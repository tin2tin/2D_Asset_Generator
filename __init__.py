bl_info = {
    "name": "2D Asset Generator (Z-Image Turbo)",
    "author": "tintwotin",
    "version": (1, 6),
    "blender": (3, 0, 0),
    "category": "3D View",
    "location": "3D Editor > Sidebar > 2D Asset",
    "description": "2D Asset Generator in the 3D View using Z-Image Turbo with Auto-Layout",
}

import bpy
from bpy.types import Operator, PropertyGroup, Panel, AddonPreferences
from bpy.props import StringProperty, EnumProperty, PointerProperty
import os, re
import subprocess
import sys
import math
import venv
import importlib
import platform
from typing import Optional

# --- MEMORY & DEVICE UTILS ---

def flush():
    """Aggressively clear RAM and VRAM before and after generation."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except:
        pass

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

DEBUG = True

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[2D Asset Gen]", *args, **kwargs)

# --- DEPENDENCY MANAGEMENT (Preserved from original) ---

def addon_script_path() -> str:
    return os.path.dirname(__file__)

def venv_path(env_name="virtual_dependencies") -> str:
    return os.path.join(addon_script_path(), env_name)

def python_exec() -> str:
    env_python = os.path.join(venv_path(), 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(venv_path(), 'bin', 'python')
    return env_python if os.path.exists(env_python) else sys.executable

def create_venv(env_name="virtual_dependencies"):
    env_dir = venv_path(env_name)
    if not os.path.exists(env_dir):
        venv.create(env_dir, with_pip=True)
        ensure_pip_installed()

def ensure_pip_installed():
    subprocess.run([python_exec(), '-m', 'ensurepip', "--upgrade"], capture_output=True)

def add_virtualenv_to_syspath():
    env_dir = venv_path()
    if platform.system() == 'Windows':
        site_packages = os.path.join(env_dir, 'lib', 'site-packages')
    else:
        site_packages = os.path.join(env_dir, 'lib', f"python{sys.version_info.major}.{sys.version_info.minor}", 'site-packages')
    
    if os.path.exists(site_packages) and site_packages not in sys.path:
        sys.path.insert(0, site_packages)

def install_packages(override: Optional[bool] = False):
    create_venv()
    add_virtualenv_to_syspath()
    python_exe = python_exec()
    
    # Requirement list for Z-Image Turbo & Image Processing
    pkgs = [
        "git+https://github.com/huggingface/diffusers.git",
        "transformers", "accelerate", "Pillow", "scipy", "numpy<2.0.0", "tqdm"
    ]
    
    for pkg in pkgs:
        subprocess.run([python_exe, "-m", "pip", "install", pkg, "--upgrade"], check=True)
    
    if gfx_device() == "cuda":
        subprocess.run([python_exe, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124"], check=True)

def check_dependencies_installed() -> bool:
    try:
        import diffusers, torch, PIL, scipy
        return True
    except ImportError:
        return False

# --- UI PROPERTIES ---

def texts_items(self, context):
    return [(text.name, text.name, "") for text in bpy.data.texts]

class Import_Text_Props(PropertyGroup):
    def update_text_list(self, context):
        if self.scene_texts in bpy.data.texts:
            self.script = bpy.data.texts[self.scene_texts].name

    input_type: EnumProperty(
        name="Input Type",
        items=[
            ("PROMPT", "Prompt", "Input: Typed in prompt"),
            ("TEXT_BLOCK", "Text-Block", "Input: Text from the Blender Text Editor"),
        ],
        default="PROMPT",
    )
    script: StringProperty(default="")
    scene_texts: EnumProperty(name="Text-Blocks", items=texts_items, update=update_text_list)

# --- ASSET NAMING UTILS ---

def get_unique_asset_name(self, context):
    base_name = context.scene.asset_name
    if not base_name:
        prompt = context.scene.asset_prompt
        base_name = "_".join(prompt.split()[:2]) if prompt else "Asset"
    
    existing_names = {obj.name for obj in bpy.data.objects}
    if base_name in existing_names:
        match = re.search(r"\((\d+)\)$", base_name)
        if match:
            base_name = base_name[: match.start()].strip()
            counter = int(match.group(1)) + 1
        else:
            counter = 1
        unique_name = f"{base_name} ({counter})"
        while unique_name in existing_names:
            counter += 1
            unique_name = f"{base_name} ({counter})"
        context.scene.asset_name = unique_name

def get_unique_file_name(base_path):
    if not os.path.exists(base_path):
        return base_path
    base_name, extension = os.path.splitext(base_path)
    counter = 1
    unique_path = f"{base_name}_{counter}{extension}"
    while os.path.exists(unique_path):
        counter += 1
        unique_path = f"{base_name}_{counter}{extension}"
    return unique_path

# --- CORE LOGIC ---

class ZIMAGE_OT_GenerateAsset(Operator):
    bl_idname = "object.generate_asset"
    bl_label = "Generate Asset"
    bl_options = {"REGISTER", "UNDO"}

    def load_model(self, context):
        import torch
        from diffusers import ZImagePipeline
        model_id = "Tongyi-MAI/Z-Image-Turbo"
        pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pipe.to(gfx_device())
        return pipe

    def generate_image(self, context, description, pipe):
        import torch
        # Z-Image Turbo inference
        out = pipe(
            prompt="neutral background, " + description,
            height=1024,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0.0,
        ).images[0]

        asset_name = re.sub(r'[<>:"/\\|?*]', "", context.scene.asset_name)
        data_path = os.path.join(bpy.utils.user_resource("DATAFILES"), "2D_Asset_Gen")
        os.makedirs(data_path, exist_ok=True)
        
        image_path = os.path.join(data_path, f"{asset_name}_gen.png")
        image_path = get_unique_file_name(image_path)
        out.save(image_path)
        return image_path

    def remove_background(self, image_path):
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms
        from PIL import Image
        import torch

        birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        birefnet.to(gfx_device())

        transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        image = Image.open(image_path).convert("RGB")
        input_image = transform_image(image).unsqueeze(0).to(gfx_device())

        with torch.no_grad():
            preds = birefnet(input_image)[-1].sigmoid().cpu()
        
        mask = transforms.ToPILImage()(preds[0].squeeze()).resize(image.size)
        image.putalpha(mask)
        
        transparent_path = image_path.replace(".png", "_rgba.png")
        image.save(transparent_path)
        return transparent_path

    def split_islands(self, image_path, output_prefix):
        from PIL import Image
        import numpy as np
        from scipy.ndimage import label, find_objects
        
        img = Image.open(image_path).convert("RGBA")
        alpha_mask = np.array(img)[:, :, 3] > 0
        labeled, num_features = label(alpha_mask)
        
        saved_paths = []
        for i, bbox in enumerate(find_objects(labeled), start=1):
            if bbox:
                crop = img.crop((bbox[1].start, bbox[0].start, bbox[1].stop, bbox[0].stop))
                file_path = os.path.join(os.path.dirname(image_path), f"{output_prefix}_{i}.png")
                file_path = get_unique_file_name(file_path)
                crop.save(file_path)
                saved_paths.append(file_path)
        return saved_paths

    def convert_to_3d(self, context, image_path, prompt, offset_x):
        from PIL import Image
        img_pil = Image.open(image_path)
        width, height = img_pil.size
        aspect = width / height
        
        # Determine spawn location based on 3D cursor + accumulated offset
        spawn_loc = context.scene.cursor.location.copy()
        spawn_loc.x += offset_x + (aspect / 2.0) # Center the plane relative to its width

        bpy.ops.mesh.primitive_plane_add(size=1, location=spawn_loc)
        obj = context.object
        obj.name = context.scene.asset_name
        obj.scale = (aspect, 1, 1)

        mat = bpy.data.materials.new(name=f"Mat_{obj.name}")
        mat.use_nodes = True
        mat.blend_method = 'HASHED'
        
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(image_path)
        
        # Shader connections (Compatibility for 4.0+)
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        mat.node_tree.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
        
        if obj.data.materials: obj.data.materials[0] = mat
        else: obj.data.materials.append(mat)

        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        obj.asset_mark()
        obj.asset_data.description = prompt
        
        # Thumbnail fix
        context.view_layer.update()
        with context.temp_override(id=obj):
            bpy.ops.ed.lib_id_load_custom_preview(filepath=image_path)
        
        return aspect # Return width for the next offset calculation

    def execute(self, context):
        add_virtualenv_to_syspath()
        if not check_dependencies_installed():
            self.report({"ERROR"}, "Dependencies not installed! Check Preferences.")
            return {"CANCELLED"}
        
        flush()
        pipe = self.load_model(context)
        
        import_type = context.scene.import_text.input_type
        if import_type == "TEXT_BLOCK":
            text = bpy.data.texts.get(context.scene.import_text.scene_texts)
            lines = [l.body for l in text.lines if l.body.strip()] if text else []
        else:
            lines = [context.scene.asset_prompt]

        # Tracking X offset to prevent overlap
        current_offset_x = 0.0
        padding = 0.2 # 20cm gap between objects

        for line in lines:
            if not line: continue
            context.scene.asset_prompt = line
            get_unique_asset_name(self, context)
            
            img_path = self.generate_image(context, line, pipe)
            rgba_path = self.remove_background(img_path)
            islands = self.split_islands(rgba_path, context.scene.asset_name)
            
            for path in islands:
                obj_width = self.convert_to_3d(context, path, line, current_offset_x)
                current_offset_x += obj_width + padding
        
        del pipe
        flush()
        return {"FINISHED"}

# --- ORIGINAL INTERFACE & PREFERENCES ---

class AssetGeneratorPreferences(AddonPreferences):
    bl_idname = __name__
    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator("virtual_dependencies.install_dependencies")
        row.operator("virtual_dependencies.check_dependencies")
        row.operator("virtual_dependencies.uninstall_dependencies")

class InstallDependenciesOperator(Operator):
    bl_idname = "virtual_dependencies.install_dependencies"
    bl_label = "Install Dependencies"
    def execute(self, context):
        install_packages(override=True)
        return {'FINISHED'}

class UninstallDependenciesOperator(Operator):
    bl_idname = "virtual_dependencies.uninstall_dependencies"
    bl_label = "Uninstall Dependencies"
    def execute(self, context):
        # Implementation to remove venv folder manually or via shutil
        self.report({'INFO'}, "Please delete the 'virtual_dependencies' folder in the addon directory.")
        return {'FINISHED'}

class CheckDependenciesOperator(Operator):
    bl_idname = "virtual_dependencies.check_dependencies"
    bl_label = "Check Dependencies"
    def execute(self, context):
        if check_dependencies_installed():
            self.report({'INFO'}, "All dependencies found and importable.")
        else:
            self.report({'ERROR'}, "Dependencies missing or broken.")
        return {'FINISHED'}

class ZIMAGE_PT_Panel(Panel):
    bl_label = "2D Asset Generator"
    bl_idname = "VIEW3D_PT_zimage_generator"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "2D Asset"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        import_text = scene.import_text

        layout = layout.box()
        row = layout.row()
        row.prop(import_text, "input_type", expand=True)

        if import_text.input_type == "TEXT_BLOCK":
            row = layout.row(align=True)
            row.prop(import_text, "scene_texts", text="", icon="TEXT")
            row.prop(import_text, "script", text="")
        else:
            layout.prop(scene, "asset_prompt", text="Prompt")
            layout.prop(scene, "asset_name", text="Name")

        layout.operator("object.generate_asset", text="Generate")

# --- REGISTRATION ---

classes = (
    Import_Text_Props,
    AssetGeneratorPreferences,
    ZIMAGE_OT_GenerateAsset,
    ZIMAGE_PT_Panel,
    InstallDependenciesOperator,
    UninstallDependenciesOperator,
    CheckDependenciesOperator,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.import_text = PointerProperty(type=Import_Text_Props)
    bpy.types.Scene.asset_prompt = StringProperty(
        name="Asset Description", 
        default="Funny monster character sheet, multiple poses, simple cartoon style, flat colors, white background"
    )
    bpy.types.Scene.asset_name = StringProperty(name="Asset Name", default="Asset", update=get_unique_asset_name)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.import_text
    del bpy.types.Scene.asset_prompt
    del bpy.types.Scene.asset_name

if __name__ == "__main__":
    register()
