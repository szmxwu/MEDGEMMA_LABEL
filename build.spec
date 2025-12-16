# build.spec
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[
        ('dcm2niix.exe', '.'),  # 确保外部工具被打包
    ],
    datas=[
        ('keywords.json', '.'),
        ('static', 'static'),
        ('templates', 'templates'),
    ],
    hiddenimports=[
        'engineio.async_drivers.gevent',
        'engineio.async_drivers.threading',
        'pynetdicom',
        'pydicom',
        'nibabel',
        'flask_socketio',
        'gevent',
        'geventwebsocket',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DICOM_WebApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # 若有GUI可改为False
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)