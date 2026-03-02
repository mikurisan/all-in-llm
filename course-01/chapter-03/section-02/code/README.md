## 1 安装 visual_bge 模块

进入 `./visual_bge` 下 执行:

```shell
pip install -e .
```

> `-e` 是 `--editable` 的意思, 将当前 module 以 editable mode 的方式 install, 从而可以本地修改该 module 而无需重新 install. 实际上只是创建了一个指向该 module 的链接, 而并没有真正 install 该 module.

在当前目录执行:

```shell
python download_model.py
```

将 model 下载到 `./model` 中.


