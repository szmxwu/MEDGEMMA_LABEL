# DICOM 下载与处理客户端

这是一个统一的 DICOM 文件下载和处理工具，可以直接从 PACS 服务器下载数据，并进行元数据提取和格式转换。

## 功能特点

- **PACS 直接集成**: 使用 DICOM 协议 (C-FIND, C-MOVE) 直接与 PACS 服务器通信。
- **元数据提取**: 将 DICOM 标签提取到 Excel 文件中。支持不同模态（MR, CT, DX, MG）的自定义模板。
- **图像转换**: 将 DICOM 序列转换为 NIfTI 格式。
- **Web 界面**: 提供友好的 Web 界面用于查询患者和管理任务。
- **多模态支持**: 针对 MRI、CT、数字X光 (DX/DR) 和乳腺钼靶 (MG) 提供专门的元数据提取支持。

## 安装说明

1. 克隆项目代码。
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```

## 配置说明

### PACS 连接配置
在项目根目录下创建 `.env` 文件，并填入 PACS 服务器信息：

```ini
# DICOM Server Configuration
PACS_IP=172.17.250.192
PACS_PORT=2104
CALLING_AET=WMX01
CALLED_AET=pacsFIR
CALLING_PORT=1103
```

### 元数据模板配置
您可以通过编辑 `dicom_tags/` 目录下的 JSON 文件来自定义不同模态提取的 DICOM 标签：
- `mr.json`: 磁共振 (MRI)
- `ct.json`: CT
- `dx.json`: 数字X光 (DR/DX/CR)
- `mg.json`: 乳腺钼靶 (Mammography)

## 使用方法

1. 启动 Web 应用：
   ```bash
   python app.py
   ```
2. 打开浏览器访问 `http://localhost:5000`。
3. 使用界面查询患者并开始下载/处理任务。

## 项目结构

- `app.py`: Flask Web 应用程序入口。
- `dicom_client_unified.py`: 核心 DICOM 处理逻辑。
- `dicom_tags/`: 元数据提取配置文件目录。
- `templates/`: Web 界面 HTML 模板。
- `static/`: 静态资源 (CSS, JS)。
- `uploads/`: 上传文件目录。
- `results/`: 处理结果目录。
