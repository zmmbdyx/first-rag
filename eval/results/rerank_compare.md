***REMOVED*** Reranker 开关对比（同索引、同题集，纯检索）

题集：191 道可回答题（single+multi_hop+paraphrase+adversarial）；k=5

| 配置 | R@1 | R@3 | R@5 | MRR | 均耗时/题 |
|---|---|---|---|---|---|
| 重排关闭 | 78.5% | 95.8% | 99.0% | 0.870 | 198.9ms |
| 重排开启 | 87.4% | 100.0% | 100.0% | 0.933 | 1432.6ms |

***REMOVED******REMOVED*** 分题型 R@5

| 题型 | 关闭 | 开启 |
|---|---|---|
| adversarial | 100% | 100% |
| multi_hop | 100% | 100% |
| paraphrase | 88% | 100% |
| single | 99% | 100% |