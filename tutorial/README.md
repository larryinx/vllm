# vLLM 中文教程

以问答形式阅读 vLLM 源码，每一讲是一个可以直接用浏览器打开的 HTML 页面。

| 讲次 | 问题 | 页面 |
| --- | --- | --- |
| 1 | 全注意力 + RoPE 模型的 KV cache 怎么存？K 存的是旋转前还是旋转后？ | [01-rope-kv-cache.html](01-rope-kv-cache.html) |
| 2 | Qwen3.5 的前向计算，以及 Gated DeltaNet 的状态缓存如何设计？ | [02-qwen3_5-gdn-cache.html](02-qwen3_5-gdn-cache.html) |
| 3 | 关闭前缀缓存时，agent 每轮工具结果回来是否整段重新 prefill？前缀缓存如何工作、哪些场景命中？ | [03-prefix-cache-agent.html](03-prefix-cache-agent.html)（模拟程序：[03-prefix-cache-sim.py](03-prefix-cache-sim.py)） |
