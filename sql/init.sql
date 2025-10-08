-- Dumped from database version 17.6
-- Dumped by pg_dump version 17.6

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
SET default_table_access_method = heap;

--
-- TOC entry 222 (class 1259 OID 16774)
-- Name: challenge_staging; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.challenge_staging (
    id bigint NOT NULL,
    file_content jsonb NOT NULL,
    file_path character varying(1024) NOT NULL,
    json_created_at timestamp without time zone NOT NULL,
    staging_inserted_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- TOC entry 221 (class 1259 OID 16773)
-- Name: challenge_staging_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.challenge_staging_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 4475 (class 0 OID 0)
-- Dependencies: 221
-- Name: challenge_staging_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.challenge_staging_id_seq OWNED BY public.challenge_staging.id;


--
-- TOC entry 224 (class 1259 OID 16784)
-- Name: challenges; Type: TABLE; Schema: public; Owner: -
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


--
-- TOC entry 223 (class 1259 OID 16783)
-- Name: challenges_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.challenges_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 4476 (class 0 OID 0)
-- Dependencies: 223
-- Name: challenges_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.challenges_id_seq OWNED BY public.challenges.id;


--
-- TOC entry 218 (class 1259 OID 16553)
-- Name: scoring_staging; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.scoring_staging (
    id bigint NOT NULL,
    file_content jsonb NOT NULL,
    file_path character varying(1024) NOT NULL,
    json_created_at timestamp without time zone NOT NULL,
    staging_inserted_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- TOC entry 217 (class 1259 OID 16552)
-- Name: scoring_staging_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.scoring_staging_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 4477 (class 0 OID 0)
-- Dependencies: 217
-- Name: scoring_staging_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.scoring_staging_id_seq OWNED BY public.scoring_staging.id;


--
-- TOC entry 220 (class 1259 OID 16623)
-- Name: scoring_submissions; Type: TABLE; Schema: public; Owner: -
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


--
-- TOC entry 219 (class 1259 OID 16622)
-- Name: scoring_submissions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.scoring_submissions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 4478 (class 0 OID 0)
-- Dependencies: 219
-- Name: scoring_submissions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.scoring_submissions_id_seq OWNED BY public.scoring_submissions.id;


--
-- TOC entry 4293 (class 2604 OID 16777)
-- Name: challenge_staging id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.challenge_staging ALTER COLUMN id SET DEFAULT nextval('public.challenge_staging_id_seq'::regclass);


--
-- TOC entry 4295 (class 2604 OID 16787)
-- Name: challenges id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.challenges ALTER COLUMN id SET DEFAULT nextval('public.challenges_id_seq'::regclass);


--
-- TOC entry 4289 (class 2604 OID 16556)
-- Name: scoring_staging id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_staging ALTER COLUMN id SET DEFAULT nextval('public.scoring_staging_id_seq'::regclass);


--
-- TOC entry 4291 (class 2604 OID 16626)
-- Name: scoring_submissions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_submissions ALTER COLUMN id SET DEFAULT nextval('public.scoring_submissions_id_seq'::regclass);


--
-- TOC entry 4315 (class 2606 OID 16782)
-- Name: challenge_staging challenge_staging_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.challenge_staging
    ADD CONSTRAINT challenge_staging_pkey PRIMARY KEY (id);


--
-- TOC entry 4317 (class 2606 OID 16793)
-- Name: challenges challenges_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.challenges
    ADD CONSTRAINT challenges_pkey PRIMARY KEY (id);


--
-- TOC entry 4305 (class 2606 OID 16561)
-- Name: scoring_staging scoring_staging_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_staging
    ADD CONSTRAINT scoring_staging_pkey PRIMARY KEY (id);


--
-- TOC entry 4313 (class 2606 OID 16637)
-- Name: scoring_submissions scoring_submissions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_submissions
    ADD CONSTRAINT scoring_submissions_pkey PRIMARY KEY (id);


--
-- TOC entry 4318 (class 1259 OID 16800)
-- Name: idx_challenges_challenge_uid; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_challenges_challenge_uid ON public.challenges USING btree (challenge_uid);


--
-- TOC entry 4319 (class 1259 OID 16801)
-- Name: idx_challenges_dialogue_uid; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_challenges_dialogue_uid ON public.challenges USING btree (dialogue_uid);


--
-- TOC entry 4320 (class 1259 OID 16802)
-- Name: idx_challenges_language; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_challenges_language ON public.challenges USING btree (language);


--
-- TOC entry 4321 (class 1259 OID 16799)
-- Name: idx_challenges_staging_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_challenges_staging_id ON public.challenges USING btree (staging_id);


--
-- TOC entry 4322 (class 1259 OID 16803)
-- Name: idx_challenges_utterance_number; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_challenges_utterance_number ON public.challenges USING btree (utterance_number);


--
-- TOC entry 4306 (class 1259 OID 16644)
-- Name: idx_scoring_submissions_challenge_uid; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_scoring_submissions_challenge_uid ON public.scoring_submissions USING btree (challenge_uid);


--
-- TOC entry 4307 (class 1259 OID 16645)
-- Name: idx_scoring_submissions_dialogue_uid; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_scoring_submissions_dialogue_uid ON public.scoring_submissions USING btree (dialogue_uid);


--
-- TOC entry 4308 (class 1259 OID 16647)
-- Name: idx_scoring_submissions_miner_hotkey; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_scoring_submissions_miner_hotkey ON public.scoring_submissions USING btree (miner_hotkey);


--
-- TOC entry 4309 (class 1259 OID 16646)
-- Name: idx_scoring_submissions_miner_uid; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_scoring_submissions_miner_uid ON public.scoring_submissions USING btree (miner_uid);


--
-- TOC entry 4310 (class 1259 OID 16643)
-- Name: idx_scoring_submissions_staging_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_scoring_submissions_staging_id ON public.scoring_submissions USING btree (scoring_staging_id);


--
-- TOC entry 4311 (class 1259 OID 16648)
-- Name: idx_scoring_submissions_utterance_number; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_scoring_submissions_utterance_number ON public.scoring_submissions USING btree (utterance_number);


--
-- TOC entry 4324 (class 2606 OID 16794)
-- Name: challenges challenges_staging_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.challenges
    ADD CONSTRAINT challenges_staging_id_fkey FOREIGN KEY (staging_id) REFERENCES public.challenge_staging(id) ON DELETE CASCADE;


--
-- TOC entry 4323 (class 2606 OID 16638)
-- Name: scoring_submissions scoring_submissions_scoring_staging_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_submissions
    ADD CONSTRAINT scoring_submissions_scoring_staging_id_fkey FOREIGN KEY (scoring_staging_id) REFERENCES public.scoring_staging(id) ON DELETE CASCADE;

