# -*- mode: python ; coding: utf-8 -*-

import os
import pkg_resources

onnxruntime_capi_path = pkg_resources.resource_filename('onnxruntime', 'capi')

# binaries = []
# for filename in sorted(glob.glob(os.path.join(onnxruntime_capi_path, '*'))):
#     if os.path.basename(filename).startswith("libonnxruntime"):
#         binaries += [(filename, '/onnxruntime/capi')]

# datas = []
# for filename in sorted(glob.glob(os.path.join("weights", '*'))):
#     datas += [(filename, 'weights')]

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[(os.path.join(onnxruntime_capi_path, "libonnxruntime*"), "/onnxruntime/capi")],
    datas=[('weights/*', 'weights')],
    hiddenimports=[],
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
    name='DistanceEstimation',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
