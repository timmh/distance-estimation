# -*- mode: python ; coding: utf-8 -*-

import os
import pkg_resources
import platform

version = os.getenv('GITHUB_REF_NAME', 'v0.0.0')[1:]

onnxruntime_capi_path = pkg_resources.resource_filename('onnxruntime', 'capi')

block_cipher = None

platform_binaries = []
if platform.system() == 'Windows':
    platform_binaries += [(pkg_resources.resource_filename('toga_winforms', 'libs'), os.path.join('toga_winforms', 'libs'))]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[(os.path.join(onnxruntime_capi_path, '*onnxruntime*'), os.path.join('onnxruntime', 'capi'))] + platform_binaries,
    datas=[('assets/*.png', 'assets')],
    hiddenimports=[
        "torch",
        "timm",
        "timm.models.layers",
        "gi",
        "gi.overrides",
        "gi.overrides.GObject",
        "gi.overrides.GLib",
        "gi.repository.GObject",
        "gi.repository.GLib",
        "gi.repository.GdkX11",
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

if platform.system() != 'Darwin':
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
        console=platform.system() == 'Windows',
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
        [],
        exclude_binaries=True,
        exclude_datas=True,
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

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        name='DistanceEstimation'
    )

    app = BUNDLE(
        coll,
        name='DistanceEstimation.app',
        icon='assets/icon.png',
        bundle_identifier='xyz.haucke.distance_estimation',
        version=version,
        info_plist={
            'CFBundleName': 'Distance Estimation',
            'CFBundleDisplayName': 'Distance Estimation',
            'CFBundleVersion': version,
            'CFBundleShortVersionString': version,
            'NSRequiresAquaSystemAppearance': 'No',
            'NSHighResolutionCapable': 'True',
        },
    )
