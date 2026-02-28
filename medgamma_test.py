import base64
import json
import requests
import time
from typing import Optional, Dict, Any

# ===== 配置区 =====
BASE_URL = "https://smartlab.cse.ust.hk/smartcare/api/shebd/medgemma15"
API_URL = f"{BASE_URL}/v1/chat/completions"
MODEL_ID = "/data/shebd/0_Pretrained/medgemma-1.5-4b-it"

# 测试图片：使用本地图片路径
TEST_IMAGE_PATH = "data/M21071400925/001_301_X_Forearm_a.p_preview.png"

# 如服务器开启了认证（Authorization header），在这里加；没有就留空
HEADERS = {
    "Content-Type": "application/json",
    # "Authorization": "Bearer xxx",  # 如无鉴权就注释掉
}

# 重试配置
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 5  # 重试间隔（秒）
REQUEST_DELAY = 2  # 每次请求前等待时间（秒），避免频率限制


def encode_image_to_base64(image_path: str) -> str:
    """
    将本地图片文件读取并转换为 base64 编码字符串
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def make_request_with_retry(payload: Dict[str, Any], max_retries: int = MAX_RETRIES) -> Optional[Dict[str, Any]]:
    """
    带重试机制的请求函数
    """
    for attempt in range(max_retries):
        try:
            # 每次请求前等待，避免频率限制
            if attempt > 0:
                print(f"\n== 第 {attempt + 1} 次重试，等待 {RETRY_DELAY} 秒... ==")
                time.sleep(RETRY_DELAY)
            else:
                print(f"\n== 请求前等待 {REQUEST_DELAY} 秒（避免频率限制）... ==")
                time.sleep(REQUEST_DELAY)
            
            print(f"== 发送请求（第 {attempt + 1}/{max_retries} 次）==")
            resp = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload), timeout=600)
            
            print(f"== HTTP 状态码: {resp.status_code} ==")
            
            # 尝试解析 JSON
            try:
                data = resp.json()
            except Exception as e:
                print(f"返回内容不是 JSON: {e}")
                print(resp.text)
                continue
            
            # 检查是否有错误
            if "error" in data:
                error_info = data["error"]
                error_type = error_info.get("type", "Unknown")
                error_code = error_info.get("code", "Unknown")
                error_msg = error_info.get("message", "")
                
                print(f"== 服务器返回错误 ==")
                print(f"类型: {error_type}")
                print(f"代码: {error_code}")
                print(f"消息: {error_msg}")
                
                # 如果是内部服务器错误，可以重试
                if error_type == "InternalServerError" or error_code == 500:
                    print(f"检测到服务器内部错误，将重试...")
                    continue
                else:
                    # 其他类型错误不重试
                    return data
            
            # 检查是否有正常的 choices
            if "choices" in data and len(data["choices"]) > 0:
                print("== 请求成功 ==")
                return data
            else:
                print("== 返回结构异常，缺少 choices 字段 ==")
                print(json.dumps(data, ensure_ascii=False, indent=2))
                continue
                
        except requests.exceptions.Timeout:
            print(f"== 请求超时 ==")
            continue
        except requests.exceptions.RequestException as e:
            print(f"== 请求异常: {e} ==")
            continue
        except Exception as e:
            print(f"== 未知错误: {e} ==")
            continue
    
    print(f"\n== 达到最大重试次数 {max_retries}，请求失败 ==")
    return None


def test_multimodal():
    """
    使用 OpenAI /v1/chat/completions 风格，发送一条带图片的消息。
    content 里包含 text + image_url 两种 type。
    """
    
    # 读取本地图片并转换为 base64
    print(f"== 正在读取本地图片: {TEST_IMAGE_PATH} ==")
    try:
        base64_image = encode_image_to_base64(TEST_IMAGE_PATH)
        image_url = f"data:image/jpeg;base64,{base64_image}"
        print(f"== 图片已成功编码（base64 长度: {len(base64_image)}）==")
    except Exception as e:
        print(f"== 读取图片失败: {e} ==")
        return

    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请详细的描述这张医学图片。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        "max_tokens": 256,
        "temperature": 0.2
    }

    print("=" * 60)
    print("== 请求 URL ==")
    print(API_URL)
    print("\n== 请求 payload ==")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("=" * 60)

    # 使用带重试的请求函数
    data = make_request_with_retry(payload)
    
    if data is None:
        print("\n所有重试均失败，无法获取响应")
        return
    
    print("\n== 原始返回 JSON ==")
    print(json.dumps(data, ensure_ascii=False, indent=2))

    # 尝试把模型回复抽出来打印
    try:
        content = data["choices"][0]["message"]["content"]
        print("\n" + "=" * 60)
        print("== 模型回复 ==")
        print(content)
        print("=" * 60)
    except Exception as e:
        print(f"\n无法从返回中解析出 choices[0].message.content: {e}")
        print("返回结构：")
        print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_multimodal()
