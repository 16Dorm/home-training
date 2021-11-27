# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['home_trainer_gui.py'],
             pathex=[ 
                 "C:\\Users\\NUDA\\miniconda3\\envs\\home7\\lib\\site-packages\\cv2\\config.py"
             ],
             binaries=[],
             datas=[ ('./classification_model', './classification_model'),
                     ('./data_loader','./data_loader'),
                     ('./dataset','./dataset'),
                     ('./GUI', './GUI'),
                     ('./utils','./utils'),
                     ('./Video','./Video'),
                     ('./home_trainer.py', './'),
                     ('./video_name.txt','./'),
                     ('./video_list.txt','./'),
             ],
             hiddenimports=["matplotlib.pyplot",    
                            "mediapipe",               
                            "mediapipe.solution",
                            "sklearn",
                            ],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='home_trainer_gui',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='home_trainer_gui')
