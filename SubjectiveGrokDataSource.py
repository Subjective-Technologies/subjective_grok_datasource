import os
import mimetypes
from typing import Any, Dict, List, Optional
import requests
from subjective_abstract_data_source_package import SubjectiveOnDemandDataSource
from brainboost_data_source_logger_package.BBLogger import BBLogger


class SubjectiveGrokDataSource(SubjectiveOnDemandDataSource):
    def __init__(self, name=None, session=None, dependency_data_sources=None,
                 subscribers=None, params=None):
        super().__init__(
            name=name,
            session=session,
            dependency_data_sources=dependency_data_sources,
            subscribers=subscribers,
            params=params
        )
        self.api_key = self.params.get("api_key")
        self.model = self.params.get("model", "grok-3")
        self.temperature = self.params.get("temperature")
        self.max_tokens = self.params.get("max_tokens")
        self.system_prompt = (self.params.get("system_prompt") or "").strip()
        self.api_base_url = (self.params.get("api_base_url") or "https://api.x.ai/v1").rstrip("/")
        self.timeout = self.params.get("timeout")

    def _process_message(self, message: Any) -> Any:
        """
        Process an incoming message and return a response from Grok.

        Args:
            message: The incoming message to process

        Returns:
            The response from Grok
        """
        BBLogger.log(f"Processing message: {str(message)[:100]}...")

        if isinstance(message, dict) and message.get("files"):
            return self._process_message_with_files(message)

        if not self.api_key:
            return {"error": "Missing Grok API key", "message": message}

        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Include recent conversation history (user/assistant only).
        for entry in self._conversation_history[-20:]:
            role = entry.get("role")
            if role not in ("user", "assistant"):
                continue
            content = entry.get("content")
            messages.append({"role": role, "content": str(content)})

        user_content = self._extract_message_text(message)
        if user_content:
            messages.append({"role": "user", "content": user_content})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        url = f"{self.api_base_url}/chat/completions"
        timeout = 60
        if isinstance(self.timeout, (int, float)) and self.timeout > 0:
            timeout = self.timeout

        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            return self._format_http_error(exc, response, message)
        data = response.json()

        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return data

    def _process_message_with_files(self, message: Dict[str, Any]) -> Any:
        if not self.api_key:
            return {"error": "Missing Grok API key", "message": message}

        user_text = str(message.get("content") or "")
        files = self._normalize_files(message.get("files"))

        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for entry in self._conversation_history[-20:]:
            role = entry.get("role")
            if role not in ("user", "assistant"):
                continue
            content = entry.get("content")
            if isinstance(content, dict):
                content = content.get("content") or content.get("message") or ""
            messages.append({"role": role, "content": str(content)})

        user_content = self._build_content(user_text, files)
        messages.append({"role": "user", "content": user_content})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        url = f"{self.api_base_url}/chat/completions"
        timeout = 60
        if isinstance(self.timeout, (int, float)) and self.timeout > 0:
            timeout = self.timeout

        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            return self._format_http_error(exc, response, message)
        data = response.json()

        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return data

    def _normalize_files(self, files: Any) -> List[Dict[str, Any]]:
        if not isinstance(files, list):
            return []
        normalized = []
        for item in files:
            if isinstance(item, dict):
                normalized.append(item)
        return normalized

    def _build_content(self, user_text: str, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        if user_text:
            content.append({"type": "text", "text": user_text})

        for payload in files:
            mime_type = payload.get("mime_type") or self._guess_mime_type(payload.get("name"))
            data_base64 = payload.get("data_base64")
            if mime_type and mime_type.startswith("image/") and data_base64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{data_base64}"}
                })
                continue

            text_block = self._format_file_text(payload)
            if text_block:
                content.append({"type": "text", "text": text_block})

        if not content:
            content.append({"type": "text", "text": ""})

        return content

    def _format_file_text(self, payload: Dict[str, Any]) -> str:
        name = payload.get("name") or "attachment"
        mime_type = payload.get("mime_type") or self._guess_mime_type(name) or "application/octet-stream"
        size = payload.get("size")
        text = payload.get("text")
        if isinstance(text, str) and text:
            return self._truncate_text(
                f"[Attached file: {name} | {mime_type} | {size} bytes]\n{text}"
            )

        data_base64 = payload.get("data_base64")
        if isinstance(data_base64, str) and data_base64:
            snippet = self._truncate_text(data_base64, max_chars=10000)
            return f"[Attached file: {name} | {mime_type} | {size} bytes | base64]\n{snippet}"

        return f"[Attached file: {name} | {mime_type} | {size} bytes | no content provided]"

    def _truncate_text(self, text: str, max_chars: int = 20000) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n[truncated]"

    def _guess_mime_type(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        mime_type, _ = mimetypes.guess_type(name)
        return mime_type

    def _extract_message_text(self, message: Any) -> str:
        if isinstance(message, dict):
            content = message.get("content") or message.get("message")
            if content is not None:
                return str(content)
            return ""
        return str(message)

    def _format_http_error(self, exc: requests.HTTPError, response: requests.Response, message: Any) -> Dict[str, Any]:
        payload = {"error": str(exc), "message": message}
        try:
            payload["status_code"] = response.status_code
            payload["response"] = response.json()
        except ValueError:
            payload["response"] = response.text
        if response.status_code == 403:
            payload["hint"] = "Check API key permissions and model access."
        return payload

    def get_icon(self) -> str:
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        try:
            with open(icon_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            BBLogger.log(f"Error reading icon file: {e}")
            return ""

    def get_connection_data(self) -> dict:
        """Return connection configuration metadata for Grok API."""
        return {
            "connection_type": "ON_DEMAND",
            "fields": [
                {
                    "name": "connection_name",
                    "type": "text",
                    "label": "Connection Name",
                    "required": True,
                    "description": "A friendly name for this Grok connection"
                },
                {
                    "name": "api_key",
                    "type": "password",
                    "label": "xAI API Key",
                    "required": True,
                    "description": "Your xAI API key for Grok"
                },
                {
                    "name": "model",
                    "type": "select",
                    "label": "Model",
                    "required": True,
                    "default": "grok-3",
                    "options": [
                        {"value": "grok-3", "label": "Grok 3 (Recommended)"},
                        {"value": "grok-3-fast", "label": "Grok 3 Fast"},
                        {"value": "grok-3-mini", "label": "Grok 3 Mini"},
                        {"value": "grok-3-mini-fast", "label": "Grok 3 Mini Fast"},
                        {"value": "grok-2", "label": "Grok 2"},
                        {"value": "grok-2-mini", "label": "Grok 2 Mini"},
                        {"value": "grok-vision-beta", "label": "Grok Vision Beta"}
                    ]
                },
                {
                    "name": "temperature",
                    "type": "number",
                    "label": "Temperature",
                    "required": False,
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "description": "Controls randomness: 0 = deterministic, 2 = very random"
                },
                {
                    "name": "max_tokens",
                    "type": "number",
                    "label": "Max Tokens",
                    "required": False,
                    "default": 4096,
                    "min": 1,
                    "max": 131072,
                    "description": "Maximum number of tokens in the response"
                },
                {
                    "name": "system_prompt",
                    "type": "textarea",
                    "label": "System Prompt",
                    "required": False,
                    "default": "You are a helpful assistant.",
                    "description": "Instructions that define the assistant's behavior"
                },
                {
                    "name": "api_base_url",
                    "type": "text",
                    "label": "API Base URL",
                    "required": False,
                    "default": "https://api.x.ai/v1",
                    "description": "Custom API endpoint (for proxies)"
                },
                {
                    "name": "timeout",
                    "type": "number",
                    "label": "Timeout (seconds)",
                    "required": False,
                    "default": 60,
                    "min": 10,
                    "max": 300,
                    "description": "Maximum time to wait for API response"
                }
            ]
        }
