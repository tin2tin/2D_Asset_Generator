bl_info = {
    "name": "2D Asset Generator (Z-Image Turbo)",
    "author": "tintwotin & AI Assistant",
    "version": (1, 1),
    "blender": (3, 0, 0),
    "category": "3D View",
    "location": "3D Editor > Sidebar > 2D Asset",
    "description": "Generate 2D Assets using Z-Image Turbo",
}

import bpy
import os
import re
import subprocess
import sys
import math
import venv
import importlib
import platform
from typing import Optional
from bpy.types import Operator, PropertyGroup, Panel, AddonPreferences
from bpy.props import StringProperty, EnumProperty, IntProperty

# --- UTILITIES & ENVIRONMENT ---

DEBUG = True

def debug_print(*args, **kwargs):
    if DEBUG:
        print(f"[2D Asset Gen] ", *args, **kwargs)

def gfx_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except:
        pass
    return "cpu"

def addon_script_path() -> str:
    return os.path.dirname(__file__)

def venv_path(env_name="virtual_dependencies") -> str:
    return os.path.join(addon_script_path(), env_name)

def python_exec() -> str:
    if platform.system() == "Windows":
        path = os.path.join(venv_path(), 'Scripts', 'python.exe')
    else:
        path = os.path.join(venv_path(), 'bin', 'python')
    return path if os.path.exists(path) else sys.executable

def get_site_packages():
    v_path = venv_path()
    if platform.system() == "Windows":
        return os.path.join(v_path, 'Lib', 'site-packages')
    else:
        # For Linux/macOS, we need to find the python3.x folder
        lib_path = os.path.join(v_path, 'lib')
        if os.path.exists(lib_path):
            for folder in os.listdir(lib_path):
                if folder.startswith("python"):
                    return os.path.join(lib_path, folder, 'site-packages')
    return os.path.join(v_path, 'lib', 'site-packages')

def create_venv():
    env_dir = venv_path()
    if not os.path.exists(env_dir):
        debug_print(f"Creating venv at {env_dir}...")
        venv.create(env_dir, with_pip=True)
    ensure_pip_installed()

def ensure_pip_installed():
    subprocess.run([python_exec(), '-m', 'ensurepip', "--upgrade", "--disable-pip-version-check"])

def add_virtualenv_to_syspath():
    sp_path = get_site_packages()
    if os.path.exists(sp_path) and sp_path not in sys.path:
        sys.path.insert(0, sp_path)
        debug_print(f"Added to sys.path: {sp_path}")

def install_packages(override=False):
    create_venv()
    add_virtualenv_to_syspath()
    
    python_exe = python_exec()
    target = get_site_packages()
    
    # Base requirements
    pkgs = [
        "diffusers", "transformers", "accelerate", "scipy", 
        "Pillow", "numpy==1.26.4", "requests", "tqdm"
    ]
    
    debug_print("Installing core dependencies...")
    for pkg in pkgs:
        cmd = [python_exe, "-m", "pip", "install", pkg, "--target", target, "--no-warn-script-location", "--upgrade"]
        if not override: cmd.insert(5, "--ignore-installed")
        subprocess.call(cmd)

    # PyTorch logic
    if platform.system() == "Windows":
        debug_print("Installing Torch for Windows (CUDA 12.1 support)...")
        subprocess.call([
            python_exe, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121", 
            "--target", target, "--upgrade"
        ])
    else:
        subprocess.call([python_exe, "-m", "pip", "install", "torch", "torchvision", "--target", target, "--upgrade"])

    debug_print("Dependency installation finished.")

def flush():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

# --- CORE LOGIC ---

def get_unique_file_name(base_path):
    base_name, extension = os.path.splitext(base_path)
    counter = 1
    unique_path = base_path
    while os.path.exists(unique_path):
        unique_path = f"{base_name}_{counter}{extension}"
        counter += 1
    return unique_path

class ZIMAGE_OT_GenerateAsset(bpy.types.Operator):
    bl_idname = "object.generate_asset"
    bl_label = "Generate Asset"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        add_virtualenv_to_syspath()
        
        try:
            import torch
            from diffusers import ZImagePipeline
            from PIL import Image
        except ImportError:
            self.report({"ERROR"}, "Dependencies not found. Please install them in Addon Preferences.")
            return {"CANCELLED"}

        scene = context.scene
        device = gfx_device()
        dtype = torch.bfloat16 if device != "cpu" else torch.float32

        # 1. Load Model
        try:
            debug_print(f"Loading Z-Image-Turbo on {device}...")
            pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=dtype)
            pipe.to(device)
        except Exception as e:
            self.report({"ERROR"}, f"Model Load Fail: {e}")
            return {"CANCELLED"}

        # 2. Prepare Prompts
        prompts = []
        if scene.import_text.input_type == "TEXT_BLOCK":
            text_obj = bpy.data.texts.get(scene.import_text.scene_texts)
            if text_obj:
                prompts = [line.body for line in text_obj.lines if line.body.strip()]
        else:
            prompts = [scene.asset_prompt]

        # 3. Process
        data_dir = os.path.join(bpy.utils.user_resource("DATAFILES"), "2D_Asset_Generator")
        os.makedirs(data_dir, exist_ok=True)

        for i, p_text in enumerate(prompts):
            debug_print(f"Generating {i+1}/{len(prompts)}: {p_text[:30]}...")
            
            # Generate
            generator = torch.Generator(device).manual_seed(torch.seed())
            image = pipe(
                p_text,
                height=1024,
                width=1024,
                num_inference_steps=9,
                guidance_scale=0.0,
                generator=generator
            ).images[0]

            # Save Initial
            clean_name = re.sub(r'[^\w\s-]', '', (scene.asset_name or "Asset")).strip().replace(" ", "_")
            img_path = get_unique_file_name(os.path.join(data_dir, f"{clean_name}.png"))
            image.save(img_path)

            # Background Removal (Using existing logic updated for speed)
            transparent_path = self.remove_background(img_path, device)
            
            # Split and Convert
            islands = self.split_by_alpha_islands(transparent_path, clean_name)
            for path in islands:
                self.convert_to_3d(context, path, p_text)

        flush()
        return {"FINISHED"}

    def remove_background(self, image_path, device):
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms
        from PIL import Image
        import torch

        model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        model.to(device)
        model.eval()

        input_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert("RGB")
        input_tensor = transform_image(image).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(input_tensor)[-1].sigmoid().cpu()
        
        mask = transforms.ToPILImage()(preds[0].squeeze())
        mask = mask.resize(image.size)
        
        image.putalpha(mask)
        out_path = image_path.replace(".png", "_rgba.png")
        image.save(out_path)
        return out_path

    def split_by_alpha_islands(self, image_path, prefix):
        from PIL import Image
        import numpy as np
        from scipy.ndimage import label, find_objects
        
        img = Image.open(image_path).convert("RGBA")
        data = np.array(img)
        alpha = data[:, :, 3] > 10
        
        labeled, num_features = label(alpha)
        slices = find_objects(labeled)
        
        paths = []
        for i, slc in enumerate(slices):
            if slc is None: continue
            # Basic size filter to ignore tiny noise
            if (slc[0].stop - slc[0].start) < 20 or (slc[1].stop - slc[1].start) < 20:
                continue
                
            crop = img.crop((slc[1].start, slc[0].start, slc[1].stop, slc[0].stop))
            p = image_path.replace("_rgba.png", f"_part_{i}.png")
            crop.save(p)
            paths.append(p)
        return paths

    def convert_to_3d(self, context, image_path, prompt):
        import math
        # Load Image
        bl_img = bpy.data.images.load(image_path)
        
        # Create Material
        mat = bpy.data.materials.new(name=f"Mat_{os.path.basename(image_path)}")
        mat.use_nodes = True
        mat.blend_method = 'HASHED'
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        bsdf = nodes.get("Principled BSDF")
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bl_img
        
        links.new(bsdf.inputs['Base Color'], tex.outputs['Color'])
        links.new(bsdf.inputs['Alpha'], tex.outputs['Alpha'])
        
        # Create Plane
        aspect = bl_img.size[0] / bl_img.size[1]
        bpy.ops.mesh.primitive_plane_add(size=1)
        obj = context.object
        obj.name = os.path.basename(image_path)
        obj.scale[0] = aspect
        obj.data.materials.append(mat)
        
        # Orient
        obj.rotation_euler[0] = math.radians(90)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        
        # Mark as Asset
        obj.asset_mark()
        obj.asset_data.description = prompt
        
        # Custom Preview
        with context.temp_override(id=obj):
            bpy.ops.ed.lib_id_load_custom_preview(filepath=image_path)

# --- UI ELEMENTS ---

class Import_Text_Props(PropertyGroup):
    input_type: EnumProperty(
        name="Input Type",
        items=[("PROMPT", "Prompt", ""), ("TEXT_BLOCK", "Text-Block", "")],
        default="PROMPT"
    )
    script: StringProperty(name="Script")
    scene_texts: EnumProperty(
        name="Text-Blocks",
        items=lambda self, context: [(t.name, t.name, "") for t in bpy.data.texts]
    )

class ZIMAGE_PT_Panel(Panel):
    bl_label = "2D Asset Generator (Z-Turbo)"
    bl_idname = "VIEW3D_PT_zimage_gen"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = '2D Asset'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.import_text

        col = layout.column(align=True)
        col.prop(props, "input_type", expand=True)
        
        if props.input_type == "TEXT_BLOCK":
            col.prop(props, "scene_texts", text="")
        else:
            col.prop(scene, "asset_prompt", text="Prompt")
            col.prop(scene, "asset_name", text="Base Name")

        layout.separator()
        layout.operator("object.generate_asset", icon='IMAGE_DATA')

class ZIMAGE_Preferences(AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        layout.label(text="Manage Dependencies (Requires Internet)")
        row = layout.row()
        row.operator("zimage.install_deps", text="Install/Update Dependencies")
        row.operator("zimage.check_deps", text="Check Status")

class ZIMAGE_OT_InstallDeps(Operator):
    bl_idname = "zimage.install_deps"
    bl_label = "Install Dependencies"
    def execute(self, context):
        install_packages(override=True)
        return {'FINISHED'}

class ZIMAGE_OT_CheckDeps(Operator):
    bl_idname = "zimage.check_deps"
    bl_label = "Check Dependencies"
    def execute(self, context):
        add_virtualenv_to_syspath()
        try:
            import torch, diffusers
            self.report({'INFO'}, f"Ready! Torch: {torch.__version__} | Device: {gfx_device()}")
        except:
            self.report({'ERROR'}, "Missing dependencies.")
        return {'FINISHED'}

# --- REGISTRATION ---

classes = (
    Import_Text_Props,
    ZIMAGE_OT_GenerateAsset,
    ZIMAGE_PT_Panel,
    ZIMAGE_Preferences,
    ZIMAGE_OT_InstallDeps,
    ZIMAGE_OT_CheckDeps,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.import_text = bpy.props.PointerProperty(type=Import_Text_Props)
    bpy.types.Scene.asset_prompt = StringProperty(name="Prompt", default="A cute 2D game character, sprite sheet style")
    bpy.types.Scene.asset_name = StringProperty(name="Name", default="NewAsset")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.import_text
    del bpy.types.Scene.asset_prompt
    del bpy.types.Scene.asset_name

if __name__ == "__main__":
    register()
