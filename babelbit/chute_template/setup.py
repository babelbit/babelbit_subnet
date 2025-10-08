def init_chute(username: str, name: str) -> Chute:
    image = (
        ChutesImage(
            username=username,
            name=name,
            tag="latest",
        )
        .from_base("parachutes/python:3.12")
        .run_command("pip install --upgrade setuptools wheel")
        .run_command(
            "pip install huggingface_hub==0.19.4 torch torchvision torchaudio")
        .run_command(
            "pip install transformers pydantic chutes"
        )
        .set_workdir("/app")
    )

    node_selector = NodeSelector(
        gpu_count=1,
        min_vram_gb_per_gpu=16,  # Minimum required by Chutes platform
    )
    return Chute(
        username=username,
        name=name,
        image=image,
        node_selector=node_selector,
        concurrency=4,
        timeout_seconds=300,
        shutdown_after_seconds=3600,
    )
