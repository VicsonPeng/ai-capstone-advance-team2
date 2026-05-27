from pxr import Usd, UsdPhysics

files = {
    "apple/model_FakeFruit_5404E_RomeRedApple_69323.usd": 0.03,
    "drink/model_drink002.usd": 0.03,
    "snack/model_snack012.usd": 0.03,
}

for path, mass in files.items():
    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim()
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.GetMassAttr().Set(mass)
    stage.GetRootLayer().Save()
    print(f"Done: {path} mass={mass}")