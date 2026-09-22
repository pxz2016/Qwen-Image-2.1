# Prompt Rewrite 双服务 Docker Compose 部署

本指南用仓库根目录的 `compose.yaml` 启动两个 OpenAI-compatible vLLM 服务：T2I 和 edit。它复用 `serve.sh`，固定使用与 `requirements.txt` 中 `vllm==0.19.1` 对应的 `vllm/vllm-openai:v0.19.1` 镜像。也可以只准备 T2I 权重并单独启动 T2I。

配置没有在 NVIDIA H20 上实测，不保证初始化时间、吞吐量或显存一定足够。请依据实际机器调整。

## 前提与资源规划

- Linux、可工作的 NVIDIA 驱动、Docker Engine、Docker Compose v2，以及 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。
- 在仓库根目录执行全部命令：

  ```bash
  cd /path/to/Qwen-Image-2.1
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
  docker compose version
  docker pull vllm/vllm-openai:v0.19.1
  docker run --rm --gpus all --entrypoint nvidia-smi vllm/vllm-openai:v0.19.1
  ```

  最后一条仅验证官方镜像的 GPU 透传，不会验证模型推理。
- 默认是**单卡共享**：两个服务都选择宿主机 GPU `0`，各自 `MEM_UTIL=0.45`。这是起步预算；两个 vLLM 实例不会自动协调，不能在同卡把两者都设为 `0.90`。请检查实际总显存和空闲显存；`0.45` 不保证显存或吞吐足够。
- 推荐的双卡方案是一服务独占一张卡，分别使用 `T2I_GPU=0`、`EDIT_GPU=1`、`T2I_MEM_UTIL=0.85`、`EDIT_MEM_UTIL=0.85`。卡数、显存和并发需要自行实测。

## 准备权重和环境文件

Compose 只读取本地目录，与下载平台无关。双服务的目录布局应为：

```text
models/
├── Qwen-Image-2.1-PE-T2I/
└── Qwen-Image-2.1-PE-I2I/
```

这两个是微调后的 prompt-enhancer 权重，不能用基础模型替代，也不能相互替代。只启动 T2I 时，仅需第一个目录。每个模型目录应直接包含 `config.json`、完整权重及 tokenizer 等模型文件，不要再嵌套一层模型目录。

选择下面一种方式准备权重即可，不需要在两个平台重复下载。模型必须在所选平台可访问；若模型未发布或账户无权限，不能继续下载该模型。已有完整权重时可直接复用。

### 方式一：ModelScope 下载（适用于已使用 modelscope 的环境）

在宿主机已安装 ModelScope CLI 的 Python 环境中执行；这些下载命令不在 vLLM 服务容器内执行，也不需要设置 `HF_TOKEN`。CLI 参数说明见 [ModelScope 官方文档](https://github.com/modelscope/modelscope/blob/master/docs/source/command.md)。

下载 T2I，并用 `--local_dir` 明确指定 Compose 使用的目录：

```bash
mkdir -p models
modelscope download \
  --model Qwen/Qwen-Image-2.1-PE-T2I \
  --local_dir ./models/Qwen-Image-2.1-PE-T2I
```

如果你已经执行过 `modelscope download --model Qwen/Qwen-Image-2.1-PE-T2I`，先查看下载日志中的实际保存路径，不要假定默认缓存位于仓库的 `models/` 下。将下面的源路径替换为实际的完整模型目录后复制，无需重新下载：

```bash
mkdir -p ./models/Qwen-Image-2.1-PE-T2I
cp -aL /absolute/path/to/downloaded/Qwen-Image-2.1-PE-T2I/. \
  ./models/Qwen-Image-2.1-PE-T2I/
```

请使用空的目标模型目录，避免混合不同版本的文件。`cp -aL` 会复制符号链接指向的实际内容；不要只创建指向宿主机缓存的符号链接，因为 Compose 仅挂载 `./models`，容器可能无法访问挂载目录外的链接目标。

仅在需要 edit 服务、且 ModelScope 上对应模型可访问时，再下载 I2I：

```bash
modelscope download \
  --model Qwen/Qwen-Image-2.1-PE-I2I \
  --local_dir ./models/Qwen-Image-2.1-PE-I2I
```

若该平台无法下载 I2I，可使用下面的 Hugging Face 方式或已有的完整 I2I 权重；不要用 T2I 权重替代。

### 方式二：Hugging Face 下载（可选）

已用 ModelScope 下载完整权重的用户跳过本节。如需通过 Hugging Face 下载，先在 Bash 终端安全地设置 `HF_TOKEN`（不要写入 `.env`、命令历史或日志），然后用镜像自带的 Python 和 `huggingface_hub` 下载。以下命令是手动准备步骤，不会在服务启动时自动执行：

```bash
mkdir -p models
read -rs HF_TOKEN; export HF_TOKEN; echo
docker run --rm -i -e HF_TOKEN -v "$PWD/models:/models" \
  --entrypoint python vllm/vllm-openai:v0.19.1 - <<'PY'
from huggingface_hub import snapshot_download
for name in ("Qwen-Image-2.1-PE-T2I", "Qwen-Image-2.1-PE-I2I"):
    snapshot_download(repo_id=f"Qwen/{name}", local_dir=f"/models/{name}")
PY
unset HF_TOKEN
```

此示例下载两个模型；若仅需要 T2I，将循环中的元组改为 `("Qwen-Image-2.1-PE-T2I",)`。不要重复下载已准备好的模型。

### 创建本地配置

仅在 `.env` 不存��时复制示例，避免覆盖已有配置。仓库忽略 `.env` 和本地权重；不要把 token 写入配置或提交到仓库。

```bash
if [ ! -e .env ]; then
  cp .env.example .env
fi
```

`.env.example` 包含清楚命名的 GPU、显存预算、上下文、图片数和 EAGER 变量。双卡独占时编辑 `.env` 为上述双卡四项。`T2I_GPU`/`EDIT_GPU` 是**宿主机** GPU ID；每个容器只分配一张 GPU，容器内保持 `GPUS=0`、`TP=1`，不要将宿主机 GPU ID 填入容器的 `GPUS`。

## 校验和启动

先渲染配置，确认路径、端口和 GPU 选择：

```bash
docker compose config
# 可选：临时验证双卡覆盖，不修改 .env
T2I_GPU=0 EDIT_GPU=1 T2I_MEM_UTIL=0.85 EDIT_MEM_UTIL=0.85 docker compose config
```

### 仅启动 T2I

只下载了 `Qwen-Image-2.1-PE-T2I` 时使用本节，不需要 I2I 权重。先检查目录中的配置文件；此检查不能代替完整权重校验或实际推理测试。

```bash
test -f ./models/Qwen-Image-2.1-PE-T2I/config.json
docker compose up -d --wait --wait-timeout 1800 t2i
docker compose ps
docker compose logs --tail=100 t2i
curl -fsS http://127.0.0.1:8100/health && echo " t2i ready"
curl -fsS http://127.0.0.1:8100/v1/models
```

未准备 I2I 权重时，不要运行不带服务名�� `docker compose up -d`，也不要启动 `edit`。指定 `t2i` 不会停止之前已运行的 edit；若需要关闭它，执行 `docker compose stop edit`。仅运行 T2I 时仍沿用默认 `T2I_MEM_UTIL=0.45`，需要根据实际显存和负载自行调整，不会自动提高预算。

### 启动双服务

确认两个模型均完整后，依次启动以避免同时加载大权重：

```bash
test -f ./models/Qwen-Image-2.1-PE-T2I/config.json
test -f ./models/Qwen-Image-2.1-PE-I2I/config.json
docker compose up -d --wait --wait-timeout 1800 t2i
docker compose up -d --wait --wait-timeout 1800 edit
docker compose ps
docker compose logs --tail=100 t2i edit
curl -fsS http://127.0.0.1:8100/health && echo " t2i ready"
curl -fsS http://127.0.0.1:8101/health && echo " edit ready"
curl -fsS http://127.0.0.1:8100/v1/models
curl -fsS http://127.0.0.1:8101/v1/models
```

`--wait` 需要支持该选项的 Compose v2；不支持时用 `docker compose up -d t2i`，轮询 `docker compose ps`/`curl` 和 `docker compose logs -f t2i`，确认健康后再按需对 edit 重复。健康检查通过仅代表服务就绪，还需要执行下面的客户端推理验证。

修改 `.env` 后需要重建对应服务，而不只是 restart。仅部署 T2I 时：

```bash
docker compose up -d --force-recreate --wait --wait-timeout 1800 t2i
```

双服务部署时依次重建：

```bash
docker compose up -d --force-recreate --wait --wait-timeout 1800 t2i
docker compose up -d --force-recreate --wait --wait-timeout 1800 edit
```

停止并删除本 Compose 项目的容器可执行 `docker compose down`；本地 `models/` 权重不会因此删除。

两个容器内部都监听 `8100`，宿主机仅绑定回环地址：T2I 为 `127.0.0.1:8100`，edit 为 `127.0.0.1:8101`，不会默认对公网开放。若需要对外服务，请在前面部署带认证、TLS 和访问控制的反向代理，不要直接将未保护的推理端口暴露到公网。

## 客户端验证

服务端不注入 system prompt，必须由客户端提供；`--model` 必须分别匹配服务的 `NAME`。以下命令在容器内请求 `http://127.0.0.1:8100/v1`。

T2I 验证：

```bash
docker compose exec -T t2i python client.py \
  --task t2i --url http://127.0.0.1:8100/v1 --model qwen-pe-t2i \
  --system-prompt prompts/system_prompt_t2i.txt \
  "一只在雨中弹吉他的柯基，电影感，横向构图"
```

仅在 edit 服务已启动时执行：

```bash
docker compose exec -T edit python client.py \
  --task edit --url http://127.0.0.1:8100/v1 --model qwen-pe-edit \
  --system-prompt prompts/system_prompt_edit.txt \
  --image data/images/1412128.png --max-new-tokens 16000 \
  "保持主体不变，将背景改成夕阳下的海边"
```

edit 支持重复传递 `--image`（按顺序）以测试双图。若从宿主机调用 edit，URL 是 `http://127.0.0.1:8101/v1`。HTTP 接口是 `/v1/chat/completions`，不是 `/rewrite`；`parse_ok` 是客户端解析结果，应结合实际输出检查，不能只凭健康检查判断推理成功。

输入文本、system prompt、图片 token 与输出预算之和必须不超过 `MAX_LEN`。edit 的 `16000` 只是初测覆盖，不是训练推荐值；默认 `24000` 需要有足够上下文。缩小 `MAX_LEN` 时也应相应减少客户端输出预算，并为输入文本和图片 token 留出空间。

## 故障排查

- `CKPT ... is not a directory`：检查 `models/` 布局和只读挂载。ModelScope 未指定 `--local_dir` 时，按下载日志找到实际目录并复制完整文件；宿主机缓存不会自动挂载到容器。
- 模型配置或权重文件找不到：确认 `config.json` 位于模型目录顶层、所有权重分片均已下载，且没有指向容器挂载范围外的符号链接。
- 只有 T2I 权重却启动 edit 失败：只执行 `docker compose up -d --wait --wait-timeout 1800 t2i`；需要 edit 时另行准备 I2I 权重。
- 模型找不到或响应不符合预期：检查客户端 `--model` 是否匹配 `qwen-pe-t2i`/`qwen-pe-edit`，并确认使用相应的微调权重和 system prompt。
- GPU 不可见：重新检查驱动、Container Toolkit、`docker run ... nvidia-smi` 和 `.env` 中宿主机 GPU ID。
- OOM 或 KV cache 不足：按实际显存减少上下文、并发或图片数，或改为一服务一卡；降低显存预算并不总能解决 OOM。
- CUDA graph capture 失败：设置对应的 `T2I_EAGER=1` 或 `EDIT_EAGER=1` 后重建服务。调整参数时以 `serve.sh` 实际支持的环境变量为准；本 Compose 暴露了 `MAX_LEN`、`MEM_UTIL`、`MAX_IMGS`、`EAGER` 的服务级覆盖，并固定单容器 `GPUS=0`、`TP=1`。
- `parse_ok=false`：检查权重与 system prompt 的配对、输入是否被截断，以及输出预算是否足够。
