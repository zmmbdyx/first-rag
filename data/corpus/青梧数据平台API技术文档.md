***REMOVED*** 青梧数据平台 API 技术文档（V2）

***REMOVED******REMOVED*** 一、接入准备

API基础地址为 https://api.qingwu.dev/v2。所有请求需携带X-Api-Key请求头进行鉴权。API Key在控制台的应用管理页面创建，每个应用最多创建5个Key。

***REMOVED******REMOVED*** 二、数据查询接口

查询接口为 GET /datasets/{id}/records，单次最多返回100条记录。分页参数为 page 和 page_size，默认page_size=20。响应中的 has_more 字段标识是否还有下一页。

***REMOVED******REMOVED*** 三、限流与配额

查询类接口限流为每秒50次，写入类接口为每秒10次。超出限流返回HTTP 429，响应头 Retry-After 提示等待秒数。免费版每月配额10万次调用，专业版为500万次。

***REMOVED******REMOVED*** 四、错误码

错误码40001表示API Key无效。错误码42901表示触发限流。错误码50001表示服务端内部错误。

***REMOVED******REMOVED*** 五、Webhook

Webhook回调需在5秒内返回HTTP 200，否则将重试3次，间隔为1分钟、5分钟、30分钟。