# Prompt Rewrite 双服务 Docker Compose 部署

本指南用仓库根目录的 `compose.yaml` 启动两个 OpenAI-compatible vLLM 服务：T2I 和 edit。它复用 `serve.sh`，固定使用与 `requirements.txt` 中 `vllm==0.19.1` 对应的 `vllm/vllm-openai:v0.19.1` 镜像；不在容器中重装整套 requirements，以免改变镜像的 CUDA/PyTorch 组合。

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
- 默认是**单卡共享**：两个服务都选择宿主机 GPU `0`，各自 `MEM_UTIL=0.45`。这是起步预算；两个 vLLM 实例不会自动协调，不能在同卡把两者都设为 `0.90`。用 `nvidia-smi` 确认总显存、空闲显存和其他进程。
- 推荐的双卡方案是一服务独占一张卡，分别使用 `T2I_GPU=0`、`EDIT_GPU=1`、`T2I_MEM_UTIL=0.85`、`EDIT_MEM_UTIL=0.85`。卡数、显存和并发需要自行实测。

## 准备权重和环境文件

Compose 只读取本地目录，布局应为：

```text
models/
├── Qwen-Image-2.1-PE-T2I/
└── Qwen-Image-2.1-PE-I2I/
```

这两个是微调后的 prompt-enhancer 权重，不能用基础模型替代。账户必须有相应下载权限；若模型未发布或账户无权限，下载和部署不能继续。已有完整权重时直接放入以上目录。

如需下载，先在终端安全地设置 `HF_TOKEN`（不要写入 `.env`、命令历史或日志），然后用镜像自带的 Python 和 `huggingface_hub` 下载。以下命令不会自动作为验证步骤执行，下载会占用大量空间和时间：

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

创建本地配置（仓库忽略 `.env`，不会提交 token 或权重）：

```bash
cp .env.example .env
```

`.env.example` 包含清楚命名的 GPU、显存预算、上下文、图片数和 EAGER 变量。双卡独占时编辑 `.env` 为上述双卡四项。`T2I_GPU`/`EDIT_GPU` 是**宿主机** GPU ID；每个容器仅暴露一张卡，因此容器内 `GPUS=0`、`TP=1` 是正确的，不要把 edit 的容器内 `GPUS` 改成 `1`。

## 校验和启动

先渲染配置，确认路径、端口和 GPU 选择：

```bash
docker compose config
# 可选：临时验证双卡覆盖，不修改 .env
T2I_GPU=0 EDIT_GPU=1 T2I_MEM_UTIL=0.85 EDIT_MEM_UTIL=0.85 docker compose config
```

依次启动以避免同时加载大权重：

```bash
docker compose up -d --wait --wait-timeout 1800 t2i
docker compose up -d --wait --wait-timeout 1800 edit
docker compose ps
docker compose logs --tail=100 t2i edit
curl -fsS http://127.0.0.1:8100/health && echo " t2i ready"
curl -fsS http://127.0.0.1:8101/health && echo " edit ready"
curl -fsS http://127.0.0.1:8100/v1/models
curl -fsS http://127.0.0.1:8101/v1/models
```

`--wait` 需要支持该选项的 Compose v2；不支持时用 `docker compose up -d t2i`，轮询 `docker compose ps`/`curl` 和 `docker compose logs -f t2i`，确认健康后再对 edit 重复。健康检查在各容器内访问 `localhost:8100/health`。停止或重建：

```bash
docker compose down
docker compose up -d --force-recreate t2i edit
```

两个容器内部都监听 `8100`，宿主机仅绑定回环地址：T2I 为 `127.0.0.1:8100`，edit 为 `127.0.0.1:8101`，不会默认对公网开放。若需要对外服务，请在前面部署带认证和 TLS 的反向代理。

## 客户端验证

服务端不注入 system prompt，必须由客户端提供；`--model` 必须分别匹配服务的 `NAME`。以下命令在容器内请求 `http://127.0.0.1:8100/v1`：

```bash
docker compose exec -T t2i python client.py \
  --task t2i --url http://127.0.0.1:8100/v1 --model qwen-pe-t2i \
  --system-prompt prompts/system_prompt_t2i.txt \
  "一只在雨中弹吉他的柯基，电影感，横向构图"

docker compose exec -T edit python client.py \
  --task edit --url http://127.0.0.1:8100/v1 --model qwen-pe-edit \
  --system-prompt prompts/system_prompt_edit.txt \
  --image data/images/1412128.png --max-new-tokens 16000 \
  "保持主体不变，将背景改成夕阳下的海边"
```

edit 支持重复传递 `--image`（按顺序）以测试双图。若从宿主机调用 edit，URL 是 `http://127.0.0.1:8101/v1`。HTTP 接口是 `/v1/chat/completions`，不是 `/rewrite`；`parse_ok` 和 `positive_prompt` 是 `client.py` 解析原始 OpenAI 响应后输出的字段。

输入文本、system prompt、图片 token 与输出预算之和必须不超过 `MAX_LEN`。edit 的 `16000` 只是初测覆盖，不是训练推荐值；默认 `24000` 需要有足够上下文。缩短输出预算可能截断 thinking 或 JSON。`EDIT_MAX_IMGS=2` 是本部署的资源限制，可调；长上下文、多图和并发都会提高资源需求。

## 故障排查

- `CKPT ... is not a directory`：检查 `models/` 布局和只读挂载。
- 模型找不到或响应不符合预期：检查客户端 `--model` 是否匹配 `qwen-pe-t2i`/`qwen-pe-edit`，并确认使用相应的微调权重和 system prompt。
- GPU 不可见：重新检查驱动、Container Toolkit、`docker run ... nvidia-smi` 和 `.env` 中宿主机 GPU ID。
- OOM 或 KV cache 不足：按实际显存减少上下文、并发或图片数，或改为一服务一卡；降低显存预算并不总能解决 OOM。
- CUDA graph capture 失败：设置对应的 `T2I_EAGER=1` 或 `EDIT_EAGER=1` 后重建服务。`serve.sh` 支持的相关选项仅包括 `MAX_LEN`、`MEM_UTIL`、`MAX_IMGS`、`EAGER`、`GPUS`、`TP` 和 `QUANT` 等环境变量，不要假定存在其他参数。
- `parse_ok=false`：检查权重与 system prompt 的配对、输入是否被截断，以及输出预算是否足够。
