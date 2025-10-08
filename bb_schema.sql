-- Dumped from database version 17.6
-- Dumped by pg_dump version 17.6

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

ALTER TABLE IF EXISTS ONLY public.scoring_submissions DROP CONSTRAINT IF EXISTS scoring_submissions_scoring_staging_id_fkey;
ALTER TABLE IF EXISTS ONLY public.challenges DROP CONSTRAINT IF EXISTS challenges_staging_id_fkey;
DROP INDEX IF EXISTS public.idx_scoring_submissions_utterance_number;
DROP INDEX IF EXISTS public.idx_scoring_submissions_staging_id;
DROP INDEX IF EXISTS public.idx_scoring_submissions_miner_uid;
DROP INDEX IF EXISTS public.idx_scoring_submissions_miner_hotkey;
DROP INDEX IF EXISTS public.idx_scoring_submissions_dialogue_uid;
DROP INDEX IF EXISTS public.idx_scoring_submissions_challenge_uid;
DROP INDEX IF EXISTS public.idx_challenges_utterance_number;
DROP INDEX IF EXISTS public.idx_challenges_staging_id;
DROP INDEX IF EXISTS public.idx_challenges_language;
DROP INDEX IF EXISTS public.idx_challenges_dialogue_uid;
DROP INDEX IF EXISTS public.idx_challenges_challenge_uid;
ALTER TABLE IF EXISTS ONLY public.scoring_submissions DROP CONSTRAINT IF EXISTS scoring_submissions_pkey;
ALTER TABLE IF EXISTS ONLY public.scoring_staging DROP CONSTRAINT IF EXISTS scoring_staging_pkey;
ALTER TABLE IF EXISTS ONLY public.challenges DROP CONSTRAINT IF EXISTS challenges_pkey;
ALTER TABLE IF EXISTS ONLY public.challenge_staging DROP CONSTRAINT IF EXISTS challenge_staging_pkey;
ALTER TABLE IF EXISTS public.scoring_submissions ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.scoring_staging ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.challenges ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.challenge_staging ALTER COLUMN id DROP DEFAULT;
DROP SEQUENCE IF EXISTS public.scoring_submissions_id_seq;
DROP TABLE IF EXISTS public.scoring_submissions;
DROP SEQUENCE IF EXISTS public.scoring_staging_id_seq;
DROP TABLE IF EXISTS public.scoring_staging;
DROP SEQUENCE IF EXISTS public.challenges_id_seq;
DROP TABLE IF EXISTS public.challenges;
DROP SEQUENCE IF EXISTS public.challenge_staging_id_seq;
DROP TABLE IF EXISTS public.challenge_staging;
DROP PROCEDURE IF EXISTS public.process_scoring_staging(IN start_date timestamp without time zone);
DROP PROCEDURE IF EXISTS public.process_json_staging(IN start_date timestamp without time zone);
DROP PROCEDURE IF EXISTS public.process_challenge_staging(IN start_date timestamp without time zone);
DROP SCHEMA IF EXISTS public;
--
-- TOC entry 4 (class 2615 OID 2200)
-- Name: public; Type: SCHEMA; Schema: -; Owner: pg_database_owner
--

ALTER SCHEMA public OWNER TO pg_database_owner;

--
-- TOC entry 4479 (class 0 OID 0)
-- Dependencies: 4
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: pg_database_owner
--

COMMENT ON SCHEMA public IS 'standard public schema';
--
-- TOC entry 237 (class 1255 OID 16591)
-- Name: process_challenge_staging(timestamp without time zone); Type: PROCEDURE; Schema: public; Owner: doadmin
--

CREATE PROCEDURE public.process_challenge_staging(IN start_date timestamp without time zone DEFAULT NULL::timestamp without time zone)
    LANGUAGE plpgsql
    AS $$
BEGIN
    INSERT INTO challenges (
        staging_id, challenge_uid, dialogue_uid, language, utterance_number, utterance_text,
        json_created_at, staging_inserted_at
    )
    SELECT 
        s.id AS staging_id,
        s.file_content->>'challenge_uid' AS challenge_uid,
        d.elem->>'dialogue_uid' AS dialogue_uid,
        d.elem->>'language' AS language,  -- Extract from dialogues array
        u.idx - 1 AS utterance_number,  -- 0-based index
        u.value::TEXT AS utterance_text,
        s.json_created_at,
        s.staging_inserted_at
    FROM challenge_staging s,
         jsonb_array_elements(s.file_content->'dialogues') WITH ORDINALITY AS d(elem, idx),
         jsonb_array_elements_text(d.elem->'utterances') WITH ORDINALITY AS u(value, idx)
    WHERE (start_date IS NULL OR s.staging_inserted_at >= start_date)
      AND s.id NOT IN (SELECT staging_id FROM challenges);

    RAISE NOTICE 'Processed challenge_staging rows after %', start_date;
END;
$$;

ALTER PROCEDURE public.process_challenge_staging(IN start_date timestamp without time zone) OWNER TO doadmin;

--
-- TOC entry 236 (class 1255 OID 16525)
-- Name: process_json_staging(timestamp without time zone); Type: PROCEDURE; Schema: public; Owner: doadmin
--

CREATE PROCEDURE public.process_json_staging(IN start_date timestamp without time zone DEFAULT NULL::timestamp without time zone)
    LANGUAGE plpgsql
    AS $$
BEGIN
    INSERT INTO miner_submission (
        staging_id, log_file, dialogue_uid, utterance_number, ground_truth, 
        best_step, u_best, total_steps, average_u_best_early,
        json_created_at, staging_inserted_at
    )
    SELECT 
        s.id AS staging_id,
        s.file_content->>'log_file' AS log_file,
        (s.file_content->>'dialogue_uid')::VARCHAR AS dialogue_uid,
        (elem->>'utterance_number')::INTEGER AS utterance_number,
        elem->>'ground_truth' AS ground_truth,
        (elem->>'best_step')::INTEGER AS best_step,
        (elem->>'U_best')::NUMERIC AS u_best,
        (elem->>'total_steps')::INTEGER AS total_steps,
        (s.file_content->'dialogue_summary'->>'average_U_best_early')::NUMERIC AS average_u_best_early,
        s.json_created_at,
        s.staging_inserted_at
    FROM json_staging s,
         jsonb_array_elements(s.file_content->'utterances') AS elem
    WHERE (start_date IS NULL OR s.staging_inserted_at >= start_date)
      AND s.id NOT IN (SELECT staging_id FROM miner_submission);  -- Skip already processed

    -- Optional: Log completion
    RAISE NOTICE 'Processed json_staging rows after %', start_date;
END;
$$;

ALTER PROCEDURE public.process_json_staging(IN start_date timestamp without time zone) OWNER TO doadmin;
--
-- TOC entry 238 (class 1255 OID 16592)
-- Name: process_scoring_staging(timestamp without time zone); Type: PROCEDURE; Schema: public; Owner: doadmin
--

CREATE PROCEDURE public.process_scoring_staging(IN start_date timestamp without time zone DEFAULT NULL::timestamp without time zone)
    LANGUAGE plpgsql
    AS $$
BEGIN
    INSERT INTO scoring_submissions (
        scoring_staging_id, challenge_uid, dialogue_uid, miner_uid, miner_hotkey,
        utterance_number, ground_truth, best_step, u_best, total_steps, average_u_best_early,
        json_created_at, staging_inserted_at
    )
    SELECT 
        s.id AS scoring_staging_id,
        s.file_content->>'challenge_uid' AS challenge_uid,
        s.file_content->>'dialogue_uid' AS dialogue_uid,
        (s.file_content->>'miner_uid')::INTEGER AS miner_uid,  -- From JSON; NULL if missing
        s.file_content->>'miner_hotkey' AS miner_hotkey,  -- From JSON; NULL if missing
        (elem->>'utterance_number')::INTEGER AS utterance_number,
        elem->>'ground_truth' AS ground_truth,
        (elem->>'best_step')::INTEGER AS best_step,
        (elem->>'U_best')::NUMERIC AS u_best,
        (elem->>'total_steps')::INTEGER AS total_steps,
        (s.file_content->'dialogue_summary'->>'average_U_best_early')::NUMERIC AS average_u_best_early,
        s.json_created_at,
        s.staging_inserted_at
    FROM scoring_staging s,
         jsonb_array_elements(s.file_content->'utterances') AS elem
    WHERE (start_date IS NULL OR s.staging_inserted_at >= start_date)
      AND s.id NOT IN (SELECT scoring_staging_id FROM scoring_submissions);

    RAISE NOTICE 'Processed scoring_staging rows after %', start_date;
END;
$$;

ALTER PROCEDURE public.process_scoring_staging(IN start_date timestamp without time zone) OWNER TO doadmin;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 222 (class 1259 OID 16774)
-- Name: challenge_staging; Type: TABLE; Schema: public; Owner: doadmin
--

CREATE TABLE public.challenge_staging (
    id bigint NOT NULL,
    file_content jsonb NOT NULL,
    file_path character varying(1024) NOT NULL,
    json_created_at timestamp without time zone NOT NULL,
    staging_inserted_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE public.challenge_staging OWNER TO doadmin;

--
-- TOC entry 221 (class 1259 OID 16773)
-- Name: challenge_staging_id_seq; Type: SEQUENCE; Schema: public; Owner: doadmin
--

CREATE SEQUENCE public.challenge_staging_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.challenge_staging_id_seq OWNER TO doadmin;

--
-- TOC entry 4480 (class 0 OID 0)
-- Dependencies: 221
-- Name: challenge_staging_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: doadmin
--

ALTER SEQUENCE public.challenge_staging_id_seq OWNED BY public.challenge_staging.id;

--
-- TOC entry 224 (class 1259 OID 16784)
-- Name: challenges; Type: TABLE; Schema: public; Owner: doadmin
--

CREATE TABLE public.challenges (
    id bigint NOT NULL,
    staging_id bigint NOT NULL,
    challenge_uid character varying(50) NOT NULL,
    dialogue_uid character varying(50) NOT NULL,
    language character varying(10),
    utterance_number integer NOT NULL,
    utterance_text text NOT NULL,
    json_created_at timestamp without time zone NOT NULL,
    staging_inserted_at timestamp without time zone NOT NULL,
    submission_inserted_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT challenges_utterance_number_check CHECK ((utterance_number >= 0))
);

ALTER TABLE public.challenges OWNER TO doadmin;

--
-- TOC entry 223 (class 1259 OID 16783)
-- Name: challenges_id_seq; Type: SEQUENCE; Schema: public; Owner: doadmin
--

CREATE SEQUENCE public.challenges_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.challenges_id_seq OWNER TO doadmin;

--
-- TOC entry 4481 (class 0 OID 0)
-- Dependencies: 223
-- Name: challenges_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: doadmin
--

ALTER SEQUENCE public.challenges_id_seq OWNED BY public.challenges.id;

--
-- TOC entry 218 (class 1259 OID 16553)
-- Name: scoring_staging; Type: TABLE; Schema: public; Owner: doadmin
--

CREATE TABLE public.scoring_staging (
    id bigint NOT NULL,
    file_content jsonb NOT NULL,
    file_path character varying(1024) NOT NULL,
    json_created_at timestamp without time zone NOT NULL,
    staging_inserted_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE public.scoring_staging OWNER TO doadmin;

--
-- TOC entry 217 (class 1259 OID 16552)
-- Name: scoring_staging_id_seq; Type: SEQUENCE; Schema: public; Owner: doadmin
--

CREATE SEQUENCE public.scoring_staging_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.scoring_staging_id_seq OWNER TO doadmin;

--
-- TOC entry 4482 (class 0 OID 0)
-- Dependencies: 217
-- Name: scoring_staging_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: doadmin
--

ALTER SEQUENCE public.scoring_staging_id_seq OWNED BY public.scoring_staging.id;

--
-- TOC entry 220 (class 1259 OID 16623)
-- Name: scoring_submissions; Type: TABLE; Schema: public; Owner: doadmin
--

CREATE TABLE public.scoring_submissions (
    id bigint NOT NULL,
    scoring_staging_id bigint NOT NULL,
    challenge_uid character varying(50),
    dialogue_uid character varying(50),
    miner_uid integer,
    miner_hotkey character varying(50),
    utterance_number integer NOT NULL,
    ground_truth text NOT NULL,
    best_step integer NOT NULL,
    u_best numeric(4,3) NOT NULL,
    total_steps integer NOT NULL,
    average_u_best_early numeric(4,3) NOT NULL,
    json_created_at timestamp without time zone NOT NULL,
    staging_inserted_at timestamp without time zone NOT NULL,
    submission_inserted_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT scoring_submissions_average_u_best_early_check CHECK (((average_u_best_early >= (0)::numeric) AND (average_u_best_early <= (1)::numeric))),
    CONSTRAINT scoring_submissions_best_step_check CHECK ((best_step >= 0)),
    CONSTRAINT scoring_submissions_miner_uid_check CHECK (((miner_uid >= 0) AND (miner_uid <= 9999))),
    CONSTRAINT scoring_submissions_total_steps_check CHECK ((total_steps >= 1)),
    CONSTRAINT scoring_submissions_u_best_check CHECK (((u_best >= (0)::numeric) AND (u_best <= (1)::numeric))),
    CONSTRAINT scoring_submissions_utterance_number_check CHECK ((utterance_number >= 0))
);

ALTER TABLE public.scoring_submissions OWNER TO doadmin;

--
-- TOC entry 219 (class 1259 OID 16622)
-- Name: scoring_submissions_id_seq; Type: SEQUENCE; Schema: public; Owner: doadmin
--

CREATE SEQUENCE public.scoring_submissions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.scoring_submissions_id_seq OWNER TO doadmin;

--
-- TOC entry 4483 (class 0 OID 0)
-- Dependencies: 219
-- Name: scoring_submissions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: doadmin
--

ALTER SEQUENCE public.scoring_submissions_id_seq OWNED BY public.scoring_submissions.id;

--
-- TOC entry 4297 (class 2604 OID 16777)
-- Name: challenge_staging id; Type: DEFAULT; Schema: public; Owner: doadmin
--

ALTER TABLE ONLY public.challenge_staging ALTER COLUMN id SET DEFAULT nextval('public.challenge_staging_id_seq'::regclass);

--
-- TOC entry 4299 (class 2604 OID 16787)
-- Name: challenges id; Type: DEFAULT; Schema: public; Owner: doadmin
--

ALTER TABLE ONLY public.challenges ALTER COLUMN id SET DEFAULT nextval('public.challenges_id_seq'::regclass);

--
-- TOC entry 4293 (class 2604 OID 16556)
-- Name: scoring_staging id; Type: DEFAULT; Schema: public; Owner: doadmin
--

ALTER TABLE ONLY public.scoring_staging ALTER COLUMN id SET DEFAULT nextval('public.scoring_staging_id_seq'::regclass);

--
-- TOC entry 4295 (class 2604 OID 16626)
-- Name: scoring_submissions id; Type: DEFAULT; Schema: public; Owner: doadmin
--

ALTER TABLE ONLY public.scoring_submissions ALTER COLUMN id SET DEFAULT nextval('public.scoring_submissions_id_seq'::regclass);

--
-- TOC entry 4319 (class 2606 OID 16782)
-- Name: challenge_staging challenge_staging_pkey; Type: CONSTRAINT; Schema: public; Owner: doadmin
--

ALTER TABLE ONLY public.challenge_staging
    ADD CONSTRAINT challenge_staging_pkey PRIMARY KEY (id);

--
-- TOC entry 4321 (class 2606 OID 16793)
-- Name: challenges challenges_pkey; Type: CONSTRAINT; Schema: public; Owner: doadmin
--

ALTER TABLE ONLY public.challenges
    ADD CONSTRAINT challenges_pkey PRIMARY KEY (id);


--
-- TOC entry 4309 (class 2606 OID 16561)
-- Name: scoring_staging scoring_staging_pkey; Type: CONSTRAINT; Schema: public; Owner: doadmin
--

ALTER TABLE ONLY public.scoring_staging
    ADD CONSTRAINT scoring_staging_pkey PRIMARY KEY (id);


--
-- TOC entry 4317 (class 2606 OID 16637)
-- Name: scoring_submissions scoring_submissions_pkey; Type: CONSTRAINT; Schema: public; Owner: doadmin
--

ALTER TABLE ONLY public.scoring_submissions
    ADD CONSTRAINT scoring_submissions_pkey PRIMARY KEY (id);

--
-- TOC entry 4322 (class 1259 OID 16800)
-- Name: idx_challenges_challenge_uid; Type: INDEX; Schema: public; Owner: doadmin
--

CREATE INDEX idx_challenges_challenge_uid ON public.challenges USING btree (challenge_uid);


--
-- TOC entry 4323 (class 1259 OID 16801)
-- Name: idx_challenges_dialogue_uid; Type: INDEX; Schema: public; Owner: doadmin
--

CREATE INDEX idx_challenges_dialogue_uid ON public.challenges USING btree (dialogue_uid);

--
-- TOC entry 4324 (class 1259 OID 16802)
-- Name: idx_challenges_language; Type: INDEX; Schema: public; Owner: doadmin
--

CREATE INDEX idx_challenges_language ON public.challenges USING btree (language);

--
-- TOC entry 4325 (class 1259 OID 16799)
-- Name: idx_challenges_staging_id; Type: INDEX; Schema: public; Owner: doadmin
--

CREATE INDEX idx_challenges_staging_id ON public.challenges USING btree (staging_id);

--
-- TOC entry 4326 (class 1259 OID 16803)
-- Name: idx_challenges_utterance_number; Type: INDEX; Schema: public; Owner: doadmin
--

CREATE INDEX idx_challenges_utterance_number ON public.challenges USING btree (utterance_number);

--
-- TOC entry 4310 (class 1259 OID 16644)
-- Name: idx_scoring_submissions_challenge_uid; Type: INDEX; Schema: public; Owner: doadmin
--

CREATE INDEX idx_scoring_submissions_challenge_uid ON public.scoring_submissions USING btree (challenge_uid);

--
-- TOC entry 4311 (class 1259 OID 16645)
-- Name: idx_scoring_submissions_dialogue_uid; Type: INDEX; Schema: public; Owner: doadmin
--

CREATE INDEX idx_scoring_submissions_dialogue_uid ON public.scoring_submissions USING btree (dialogue_uid);

--
-- TOC entry 4312 (class 1259 OID 16647)
-- Name: idx_scoring_submissions_miner_hotkey; Type: INDEX; Schema: public; Owner: doadmin
--

CREATE INDEX idx_scoring_submissions_miner_hotkey ON public.scoring_submissions USING btree (miner_hotkey);

--
-- TOC entry 4313 (class 1259 OID 16646)
-- Name: idx_scoring_submissions_miner_uid; Type: INDEX; Schema: public; Owner: doadmin
--

CREATE INDEX idx_scoring_submissions_miner_uid ON public.scoring_submissions USING btree (miner_uid);

--
-- TOC entry 4314 (class 1259 OID 16643)
-- Name: idx_scoring_submissions_staging_id; Type: INDEX; Schema: public; Owner: doadmin
--

CREATE INDEX idx_scoring_submissions_staging_id ON public.scoring_submissions USING btree (scoring_staging_id);

--
-- TOC entry 4315 (class 1259 OID 16648)
-- Name: idx_scoring_submissions_utterance_number; Type: INDEX; Schema: public; Owner: doadmin
--

CREATE INDEX idx_scoring_submissions_utterance_number ON public.scoring_submissions USING btree (utterance_number);

--
-- TOC entry 4328 (class 2606 OID 16794)
-- Name: challenges challenges_staging_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: doadmin
--

ALTER TABLE ONLY public.challenges
    ADD CONSTRAINT challenges_staging_id_fkey FOREIGN KEY (staging_id) REFERENCES public.challenge_staging(id) ON DELETE CASCADE;

--
-- TOC entry 4327 (class 2606 OID 16638)
-- Name: scoring_submissions scoring_submissions_scoring_staging_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: doadmin
--

ALTER TABLE ONLY public.scoring_submissions
    ADD CONSTRAINT scoring_submissions_scoring_staging_id_fkey FOREIGN KEY (scoring_staging_id) REFERENCES public.scoring_staging(id) ON DELETE CASCADE;

-- Completed on 2025-10-08 15:58:57
