## Tips

如果一直卡在 `Fetching 30 files: 0%`, 原因说是网络问题, 但是 vpn 可以正常访问, 设置了 mirror 也没用, 所以不懂了.

解决方式是通过 huggingface hub 下载:

```shell
pip install huggingface_hub
```

然后手动下载 model:

```py

from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir="./cache",
    resume_download=True
)
```

很奇怪的是, 这里 fetching 就很快.

但是呢, 代码还是无法运行, 说是 model version 太旧和 pytorch version 太新的冲突, 我真服了.