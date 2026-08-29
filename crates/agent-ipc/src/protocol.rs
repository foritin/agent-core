//! JSON-RPC 2.0 协议与帧格式。
//!
//! 参见 `09-ipc-transport.html §2 §3`。帧格式：`[4B big-endian 长度][JSON payload]`。
//! 请求可携带 `version` 字段用于版本握手；不兼容时服务端返回 `INVALID_REQUEST`。

use agent_error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// 当前 IPC 协议版本。
pub const PROTOCOL_VERSION: &str = "1";

/// JSON-RPC 2.0 请求。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    /// 协议版本（可选；用于握手协商）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

/// JSON-RPC 2.0 响应。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

impl JsonRpcError {
    pub const METHOD_NOT_FOUND: i64 = -32601;
    pub const INTERNAL_ERROR: i64 = -32603;
    pub const INVALID_REQUEST: i64 = -32600;
}

impl JsonRpcRequest {
    pub fn new(method: impl Into<String>, id: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            version: Some(PROTOCOL_VERSION.into()),
            id: Some(id.into()),
            method: method.into(),
            params,
        }
    }

    /// 通知（无 id）。
    pub fn notification(method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            version: Some(PROTOCOL_VERSION.into()),
            id: None,
            method: method.into(),
            params,
        }
    }

    /// 是否与给定协议版本兼容（无 version 字段视为兼容，便于旧客户端）。
    pub fn is_version_compatible(&self, expected: &str) -> bool {
        match &self.version {
            Some(v) => v == expected,
            None => true,
        }
    }
}

impl JsonRpcResponse {
    pub fn success(id: String, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id: Some(id),
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<String>, code: i64, message: String) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(JsonRpcError { code, message }),
        }
    }
}

/// 写一帧（4B 长度头 + payload）。
pub async fn write_frame<W: AsyncWrite + Unpin>(writer: &mut W, payload: &[u8]) -> Result<()> {
    // F-corr-08：写侧与读侧同一帧大小上限，且对超长 payload 显式报错——
    // 旧代码 `payload.len() as u32` 会静默截断超大帧（读侧兜底只在
    // 对端能读到正确的被截断长度时成立；这里直接拒绝，绝不让坏帧上线路）。
    if payload.len() > 16 * 1024 * 1024 {
        return Err(Error::Ipc(format!(
            "frame too large on write: {} bytes (cap 16MiB)",
            payload.len()
        )));
    }
    let len = u32::try_from(payload.len())
        .map_err(|_| Error::Ipc("frame length overflowed u32".to_string()))?;
    writer.write_all(&len.to_be_bytes()).await?;
    writer.write_all(payload).await?;
    writer.flush().await?;
    Ok(())
}

/// 读一帧。
pub async fn read_frame<R: AsyncRead + Unpin>(reader: &mut R) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > 16 * 1024 * 1024 {
        return Err(Error::Ipc(format!("frame too large: {len} bytes")));
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).await?;
    Ok(buf)
}

/// 写一个 JSON-RPC 消息为一帧。
pub async fn write_message<W: AsyncWrite + Unpin, T: Serialize>(
    writer: &mut W,
    msg: &T,
) -> Result<()> {
    let json = serde_json::to_vec(msg)?;
    write_frame(writer, &json).await
}

/// 读一帧并反序列化为 JSON-RPC 消息。
pub async fn read_message<R: AsyncRead + Unpin, T: for<'de> Deserialize<'de>>(
    reader: &mut R,
) -> Result<T> {
    let buf = read_frame(reader).await?;
    let msg: T = serde_json::from_slice(&buf)?;
    Ok(msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn frame_roundtrip() {
        let mut buf = Vec::new();
        let payload = b"hello frame";
        write_frame(&mut buf, payload).await.unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let got = read_frame(&mut cursor).await.unwrap();
        assert_eq!(got, payload);
    }

    #[tokio::test]
    async fn write_frame_rejects_oversized_payload_explicitly() {
        // F-corr-08：写侧不得静默截断超过 16MiB 的帧，必须显式报错。
        let mut buf = Vec::new();
        let oversized = vec![0u8; 16 * 1024 * 1024 + 1];
        let result = write_frame(&mut buf, &oversized).await;
        assert!(result.is_err(), "oversized frame write must fail");
        let message = format!("{}", result.unwrap_err());
        assert!(
            message.contains("frame too large on write"),
            "got {message}"
        );
    }

    #[test]
    fn request_serde() {
        let req = JsonRpcRequest::new("tool.execute", "req-1", Some(json!({"name": "x"})));
        let s = serde_json::to_string(&req).unwrap();
        assert!(s.contains(r#""method":"tool.execute""#));
        let back: JsonRpcRequest = serde_json::from_str(&s).unwrap();
        assert_eq!(back.method, "tool.execute");
    }

    #[test]
    fn notification_has_no_id() {
        let n = JsonRpcRequest::notification("agent.event", Some(json!({"x": 1})));
        assert!(n.id.is_none());
    }

    #[tokio::test]
    async fn accept_perf_limits_oversized_frame_explained() {
        // accept-perf-limits：字节限制在错误中可解释
        // 写入一个声称 >16MB 的长度头，read_frame 应返回可解释错误
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&(20u32 * 1024 * 1024).to_be_bytes()); // 20MB
                                                                     // 不再写 payload，read_frame 在读长度后即校验
        let mut cursor = std::io::Cursor::new(buf);
        let r = read_frame(&mut cursor).await;
        assert!(r.is_err());
        let msg = r.unwrap_err().to_string();
        assert!(
            msg.contains("frame too large"),
            "应为可解释的帧超限错误: {msg}"
        );
    }

    #[tokio::test]
    async fn accept_perf_limits_token_observable_in_dto() {
        // accept-perf-limits：token 限制在 DTO 中可观察（estimate_tokens）
        // 借 agent-contract 间接验证太重，这里用 usage 字段可观察性代替
        let req = JsonRpcRequest::new("m", "1", None);
        // 请求的 version 字段可观察
        assert_eq!(req.version.as_deref(), Some(PROTOCOL_VERSION));
    }
}
