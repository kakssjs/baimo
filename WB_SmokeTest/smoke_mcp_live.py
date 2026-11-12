"""Live test: talk to the open Blender GUI through the real mcp-for-blender MCP server (stdio),
exactly as Claude Code does. Run: uv run --with mcp python smoke_mcp_live.py"""
import asyncio, base64, os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output").replace("\\", "/")
UVX = r"C:\Users\kakssjs\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uvx.exe"
HDR = ('import sys; p=r"C:\\Users\\kakssjs\\.claude\\skills\\blender-whitebox\\scripts"\n'
       'sys.path.insert(0, p) if p not in sys.path else None\nimport whitebox as wb, bpy\n')

STEPS = {
 "1_build": HDR + """
wb.floor("Floor_Slab", (6.4, 4.4), 0.2, (3, 2, 0))
wb.wall("Wall_South", (0,0), (6,0), extend=0.1); wb.wall("Wall_North", (0,4), (6,4), extend=0.1)
wb.wall("Wall_West", (0,0), (0,4)); wb.wall("Wall_East", (6,0), (6,4))
wb.opening("Wall_South", "Door_Main", 1.2, 0.9, 2.1)
wb.opening("Wall_East", "Win_East_01", 2.0, 1.2, 1.2, sill=0.9)
wb.box("Table_Top", (1.6, 0.8, 0.04), (3, 2, 0.71))
wb.cylinder("Vase_Cyl", 0.12, 0.35, (3.4, 2.1, 0.75))
wb.rounded_box("Speaker_Rounded", (0.25,0.25,0.4), 0.04, (2.5,1.9,0.75))
wb.clay_all(); wb.preview_setup(target=(3,2,0.8), cam_loc=(8.5,-4.5,8.5), lens=24)
print("COUNT", len(bpy.data.objects))
""",
 "2_edit": HDR + """
n = len(bpy.data.objects)
wb.resize("Table_Top", (1.8, 0.9, 0.04))
wb.opening("Wall_East", "Win_East_01", 2.0, 1.5, 1.2, sill=0.9)
bpy.data.objects["Vase_Cyl"].location.x = 3.6
print("SAME_COUNT", n == len(bpy.data.objects), "TABLE", [round(d,3) for d in bpy.data.objects["Table_Top"].dimensions])
""",
 "3_save_render_export": HDR + f"""
bpy.ops.wm.save_as_mainfile(filepath="{OUT}/WB_LiveMCP_room.blend")
wb.render("{OUT}/WB_LiveMCP_preview.png")
wb.export("{OUT}/WB_LiveMCP_room.fbx")
print("SAVED", bpy.data.filepath)
""",
}

async def main():
    params = StdioServerParameters(command=UVX, args=["mcp-for-blender@2.2.7"],
        env={**os.environ, "DISABLE_TELEMETRY": "true", "UV_CACHE_DIR": r"C:\Users\kakssjs\.cache\uv"})
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            print("TOOLS", len(tools), [t for t in tools if t in ("execute_blender_code", "get_scene_info", "look", "get_viewport_screenshot")])
            for name, code in STEPS.items():
                res = await s.call_tool("execute_blender_code", {"code": code})
                print(name, "error" if res.isError else "ok", " ".join(c.text for c in res.content if c.type == "text")[:400])
            res = await s.call_tool("get_scene_info", {})
            print("SCENE_INFO", " ".join(c.text for c in res.content if c.type == "text")[:300])
            res = await s.call_tool("look", {"mode": "camera"})
            for c in res.content:
                if c.type == "image":
                    open(f"{OUT}/WB_LiveMCP_look.png", "wb").write(base64.b64decode(c.data)); print("LOOK_IMAGE saved")
                    break
            else:
                print("LOOK", [getattr(c, "text", c.type)[:200] for c in res.content])

asyncio.run(main())
