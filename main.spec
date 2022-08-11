# -*- mode: python ; coding: utf-8 -*-

import os
import pkg_resources
import platform

onnxruntime_capi_path = pkg_resources.resource_filename('onnxruntime', 'capi')

block_cipher = None

platform_binaries = []
if platform.system() == 'Windows':
    platform_binaries += [(pkg_resources.resource_filename('toga_winforms', 'libs'), os.path.join('toga_winforms', 'libs'))]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[(os.path.join(onnxruntime_capi_path, '*onnxruntime*'), os.path.join('onnxruntime', 'capi'))] + platform_binaries,
    datas=[('weights/*', 'weights'), ('assets/*', 'assets')],
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
if platform.system() != 'Darwin':
    splash = Splash(
        'assets/splash.png',
        binaries=a.binaries,
        datas=a.datas,
        text_pos=None,
        text_size=12,
        minify_script=True,
        always_on_top=True,
    )

if platform.system() != 'Darwin':
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        splash,
        splash.binaries,
        [],
        name='DistanceEstimation',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='assets/icon.ico',
    )
else:
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
        upx=False,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='assets/icon.ico',
    )

if platform.system() == 'Darwin':
    app = BUNDLE(
        exe,
        name='DistanceEstimation.app',
        icon='assets/icon.png',
        bundle_identifier='xyz.haucke.distance_estimation'
    )