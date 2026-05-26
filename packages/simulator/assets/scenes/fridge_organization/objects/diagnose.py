from pxr import Usd, UsdPhysics

# drink：移除子 prim 的 RigidBodyAPI
stage = Usd.Stage.Open("drink/model_drink002.usd")
child = stage.GetPrimAtPath("/root/E_drink_1")
if child.HasAPI(UsdPhysics.RigidBodyAPI):
    child.RemoveAPI(UsdPhysics.RigidBodyAPI)
    print("drink: removed RigidBodyAPI from E_drink_1")
stage.GetRootLayer().Save()

# snack：移除子 prim 的 RigidBodyAPI
stage = Usd.Stage.Open("snack/model_snack012.usd")
child = stage.GetPrimAtPath("/root/E_snack_1")
if child.HasAPI(UsdPhysics.RigidBodyAPI):
    child.RemoveAPI(UsdPhysics.RigidBodyAPI)
    print("snack: removed RigidBodyAPI from E_snack_1")
stage.GetRootLayer().Save()