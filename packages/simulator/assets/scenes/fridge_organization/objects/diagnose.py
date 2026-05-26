from pxr import Usd, UsdPhysics

# apple：加在 root
stage = Usd.Stage.Open("apple/model_FakeFruit_5404E_RomeRedApple_69323.usd")
root = stage.GetDefaultPrim()
UsdPhysics.RigidBodyAPI.Apply(root)
UsdPhysics.CollisionAPI.Apply(root)
stage.GetRootLayer().Save()
print("apple done")

# drink：把 RigidBodyAPI 從子 prim 移到 root
stage = Usd.Stage.Open("drink/model_drink002.usd")
root = stage.GetDefaultPrim()
child = stage.GetPrimAtPath("/root/E_drink_1")
UsdPhysics.RigidBodyAPI.Apply(root)
UsdPhysics.CollisionAPI.Apply(root)
if child.HasAPI(UsdPhysics.RigidBodyAPI):
    UsdPhysics.RigidBodyAPI.Get(stage, child.GetPath()).Apply(root)
stage.GetRootLayer().Save()
print("drink done")

# snack：把 RigidBodyAPI 從子 prim 移到 root
stage = Usd.Stage.Open("snack/model_snack012.usd")
root = stage.GetDefaultPrim()
child = stage.GetPrimAtPath("/root/E_snack_1")
UsdPhysics.RigidBodyAPI.Apply(root)
UsdPhysics.CollisionAPI.Apply(root)
stage.GetRootLayer().Save()
print("snack done")