from pxr import Usd, UsdPhysics

files = [
    "apple/model_FakeFruit_5404E_RomeRedApple_69323.usd",
    "drink/model_drink002.usd",
    "snack/model_snack012.usd",
]

for path in files:
    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim()
    if root:
        UsdPhysics.RigidBodyAPI.Apply(root)
        UsdPhysics.CollisionAPI.Apply(root)
        stage.GetRootLayer().Save()
        print(f"Done: {path}")
    else:
        print(f"ERROR: No default prim in {path}")