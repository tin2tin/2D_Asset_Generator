# <------------------------------------- START: Keep bl_info at the top ------------------------------------->
bl_info = {
    "name": "2D Asset Generator",
    "author": "tintwotin (Improved by AI Assistant)",
    "version": (1, 1, 0), # Incremented version
    "blender": (3, 6, 0), # IMPORTANT: Changed to match Python 3.11 required by wheels
    "category": "3D View",
    "location": "3D Editor > Sidebar > 2D Asset",
    "description": "Generates 2D assets from prompts, converts to 3D planes and GLB models.",
    "warning": "Requires manual dependency installation via Add-on Preferences. Uses external models.",
    "doc_url": "", # Add your documentation URL here
    "tracker_url": "", # Add your bug tracker URL here
}
# <-------------------------------------- END: Keep bl_info at the top -------------------------------------->


import bpy
from bpy.types import Operator, PropertyGroup, Panel, AddonPreferences
from bpy.props import StringProperty, EnumProperty, PointerProperty, BoolProperty
import os
import re
import subprocess
import sys
import math
from os.path import join, dirname, exists, splitext, basename, abspath
from pathlib import Path
from mathutils import Vector
import venv
import importlib
from typing import Optional, List, Tuple, Set # Added Set
import platform
import shutil
import stat
import gc # Garbage Collection

# --- Constants ---
DEBUG = True # Set to False for production release
ADDON_ID = __name__ # Use the addon's module name as its ID
VENV_DIR_NAME = "asset_generator_venv" # Name for the virtual environment directory

# --- Utility Functions --- (Keep existing utility functions)
def log_warning(*args, **kwargs):
    """Warning logging."""
    print(f"{ADDON_ID} [WARNING]:", *args, **kwargs)

def debug_print(*args, **kwargs):
    """Conditional print function based on the DEBUG variable."""
    if DEBUG:
        print(f"{ADDON_ID} [DEBUG]:", *args, **kwargs)

def log_info(*args, **kwargs):
    """Standard logging."""
    print(f"{ADDON_ID} [INFO]:", *args, **kwargs)

def log_error(*args, **kwargs):
    """Error logging."""
    print(f"{ADDON_ID} [ERROR]:", *args, **kwargs)


def get_addon_pref():
    """Get addon preferences."""
    return bpy.context.preferences.addons[ADDON_ID].preferences

def addon_root_path() -> Path:
    """Return the root path where the add-on script is located."""
    return Path(dirname(__file__))

def user_data_path() -> Path:
    """Get a persistent user data path for the addon."""
    path = Path(bpy.utils.user_resource("DATAFILES")) / f"{ADDON_ID}_data"
    path.mkdir(parents=True, exist_ok=True)
    return path

def venv_path() -> Path:
    """Define the path for the virtual environment directory in user data."""
    return user_data_path() / VENV_DIR_NAME

def python_exec_path() -> str:
    """Return the path to the Python executable in the virtual environment."""
    venv_p = venv_path()
    if platform.system() == 'Windows':
        py_exec = venv_p / 'Scripts' / 'python.exe'
    else:
        py_exec = venv_p / 'bin' / 'python'

    if py_exec.exists():
        return str(py_exec)
    else:
        log_info(f"Venv python not found at {py_exec}, falling back to Blender's Python: {sys.executable}")
        return sys.executable

def get_site_packages_path() -> Optional[Path]:
    """Get the site-packages path within the virtual environment."""
    venv_p = venv_path()
    if not venv_p.exists():
        return None

    lib_path = venv_p / 'lib'
    if not lib_path.exists() and platform.system() == 'Windows':
         lib_path = venv_p / 'Lib'

    if lib_path.exists():
        py_ver_dirs = list(lib_path.glob('python*'))
        if py_ver_dirs:
            site_packages = py_ver_dirs[0] / 'site-packages'
            if site_packages.exists():
                return site_packages

    site_packages_alt = lib_path / 'site-packages'
    if site_packages_alt.exists():
         return site_packages_alt

    log_error(f"Could not find site-packages directory in {venv_p}")
    return None

def add_venv_to_sys_path():
    """Add the virtual environment's site-packages directory to sys.path if not already present."""
    site_packages = get_site_packages_path()
    if site_packages:
        site_packages_str = str(site_packages)
        if site_packages_str not in sys.path:
            sys.path.insert(0, site_packages_str)
            debug_print(f"Added venv site-packages to sys.path: {site_packages_str}")
    else:
        debug_print("Virtual environment site-packages path not found. Cannot add to sys.path.")

def create_venv_if_needed():
    """Create a virtual environment if it doesn't exist."""
    env_dir = venv_path()
    if not env_dir.exists():
        log_info(f"Creating virtual environment at {env_dir}...")
        blender_py_exec = sys.executable
        try:
            log_info(f"Using Blender Python '{blender_py_exec}' to create venv.")
            subprocess.run([blender_py_exec, '-m', 'venv', str(env_dir)], check=True, capture_output=True)
            log_info(f"Virtual environment created successfully.")
            ensure_pip()
        except subprocess.CalledProcessError as e:
            log_error(f"Failed to create virtual environment.")
            log_error(f"Command: {e.cmd}")
            log_error(f"Return Code: {e.returncode}")
            stderr_decoded = e.stderr.decode(errors='ignore') if e.stderr else "N/A"
            stdout_decoded = e.stdout.decode(errors='ignore') if e.stdout else "N/A"
            log_error(f"Stderr: {stderr_decoded.strip()}")
            log_error(f"Stdout: {stdout_decoded.strip()}")
            raise
        except FileNotFoundError:
             log_error(f"Could not find Python executable '{blender_py_exec}' to create venv.")
             raise

def ensure_pip():
     """Ensure pip is available in the virtual environment."""
     py_exec = python_exec_path()
     try:
         subprocess.run([py_exec, '-m', 'ensurepip', '--upgrade'], check=True, capture_output=True)
         debug_print("Checked/Ensured pip is installed in venv.")
     except subprocess.CalledProcessError as e:
         log_error("Failed to ensure pip.")
         log_error(f"Command: {e.cmd}")
         log_error(f"Stderr: {e.stderr.decode()}")

def run_pip(args: List[str], error_message: str = "Pip command failed"):
    """Run a pip command using the virtual environment's Python."""
    py_exec = python_exec_path()
    full_cmd = [py_exec, '-m', 'pip'] + args
    log_info(f"Running command: {' '.join(full_cmd)}")
    try:
        env = os.environ.copy()
        env['PYTHONNOUSERSITE'] = '1'
        subprocess.run(full_cmd, check=True, capture_output=True, env=env)
        log_info("Pip command successful.")
    except subprocess.CalledProcessError as e:
        log_error(error_message)
        log_error(f"Command: {' '.join(e.cmd)}")
        log_error(f"Return Code: {e.returncode}")
        log_error(f"Stderr: {e.stderr.decode().strip()}")
        log_error(f"Stdout: {e.stdout.decode().strip()}")
        raise
    except FileNotFoundError:
        log_error(f"Could not find Python/Pip executable '{py_exec}'. Venv might be corrupted or not created.")
        raise

def run_command(command: List[str], cwd: Optional[str] = None, shell: bool = False, error_message: str = "Command failed"):
    """Run a shell command and handle errors."""
    log_info(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, shell=shell, check=True, cwd=cwd, capture_output=True, text=True)
        debug_print(f"Command stdout:\n{result.stdout}")
        log_info("Command successful.")
    except subprocess.CalledProcessError as e:
        log_error(error_message)
        log_error(f"Command: {' '.join(e.cmd)}")
        log_error(f"Return Code: {e.returncode}")
        log_error(f"Stderr:\n{e.stderr}")
        log_error(f"Stdout:\n{e.stdout}")
        raise
    except FileNotFoundError:
        log_error(f"Command not found: {command[0]}. Ensure it's installed and in PATH.")
        raise

# --- onerror handler needed for shutil.rmtree ---
def onerror_remove_readonly(func, path, exc_info):
    """Error handler for shutil.rmtree. Tries to change permissions."""
    # Check if the error is permission error
    if not os.access(path, os.W_OK):
        # Try changing the permissions
        os.chmod(path, stat.S_IWUSR)
        # Retry the function
        func(path)
    else:
        # Re-raise the error if it's not a permission issue
        # or changing permission didn't work
        raise exc_info[1]


# --- Adjust install_trellis_dependencies ---
def install_trellis_dependencies(force_reinstall: bool = False):
    """Install specific dependencies needed by Trellis, often requiring wheels or git."""
    log_info("Installing Trellis-specific dependencies...")
    addon_root = addon_root_path()
    whl_dir = addon_root / "whl"

    pip_args = ['install']
    pip_flags = ['--no-cache-dir', '--no-warn-script-location', '--disable-pip-version-check']
    if force_reinstall:
        pip_args.append('--force-reinstall')

    # Define dependencies needing special handling
    # Added bitsandbytes here as it can be tricky
    # Grouped by type for clarity
    special_deps = [
        # Git Repos (install with dependencies)
        ("utils3d", "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8", "git", "Installing utils3d from git"),
        ("diffusers", "git+https://github.com/huggingface/diffusers.git", "git", "Installing diffusers from git"),
        ("transformers", "git+https://github.com/huggingface/transformers.git", "git", "Installing transformers from git"),
        ("accelerate", "git+https://github.com/huggingface/accelerate.git", "git", "Installing accelerate from git"),

        # Pip packages (install with dependencies)
        #("spconv", "spconv-cu118==2.3.6", "pip", "Installing spconv (cu118)"), # WARNING: Still CUDA mismatch potential with cu124 torch/flash_attn!
        ("spconv", "spconv-cu124", "pip", "Installing spconv (cu124)"), # WARNING: Still CUDA mismatch potential with cu124 torch/flash_attn!
        ("bitsandbytes", "bitsandbytes", "pip", "Installing bitsandbytes"), # Try standard pip first

        # Wheels (install WITHOUT dependencies by default, assuming torch/cuda handled)
        # Make absolutely sure these filenames are correct and match python 3.11 / win_amd64
        ("nvdiffrast", str(whl_dir / "nvdiffrast-0.3.3-cp311-cp311-win_amd64.whl"), "wheel", "Installing nvdiffrast from local wheel"),
        ("diffoctreerast", str(whl_dir / "diffoctreerast-0.0.0-cp311-cp311-win_amd64.whl"), "wheel", "Installing diffoctreerast from local wheel"),
        ("diff_gaussian_rasterization", str(whl_dir / "diff_gaussian_rasterization-0.0.0-cp311-cp311-win_amd64.whl"), "wheel", "Installing diff_gaussian_rasterization from local wheel"),
        # Ensure this flash_attn wheel matches PyTorch cu124 AND your system CUDA 12.4 install
        ("flash_attn", "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.0.post2%2Bcu124torch2.4.1cxx11abiFALSE-cp311-cp311-win_amd64.whl", "wheel_url", "Installing flash_attn from URL (cu124)"),

        # Custom Copies (manual file copy, no pip involved for the package itself)
        # Ensure the subdir name ('trellis', 'api_spz') matches the actual folder in the repo root
        ("trellis", "https://github.com/microsoft/TRELLIS.git", "custom_copy", "Copying Trellis library files"),
        ("api_spz", "https://github.com/IgorAherne/trellis-stable-projectorz.git", "custom_copy", "Copying api_spz library files"),
    ]

    site_packages_dir = get_site_packages_path()
    if not site_packages_dir:
         log_error("Cannot determine site-packages path. Aborting special dependency installation.")
         raise RuntimeError("Site-packages path not found.")

    cloned_repos = {}

    for name, target, install_type, comment in special_deps:
        log_info(comment)
        clone_dir = None
        current_pip_args = pip_args[:] # Start with base args (e.g., ['install'] or ['install', '--force-reinstall'])

        # Determine dependency handling and flags
        needs_deps = install_type in ["git", "pip"] # Git/Pip installs usually need deps
        if needs_deps:
            current_pip_args.extend(pip_flags) # Add standard flags
        else: # Wheels / Custom Copy don't use pip deps flag here
            current_pip_args.extend(pip_flags) # Add standard flags
            if install_type in ["wheel", "wheel_url"]:
                 # Optionally add --no-deps if confident dependencies are met elsewhere
                 # current_pip_args.append('--no-deps') # Use with caution
                 pass # Wheels often bundle or expect system libs

        try:
            if install_type == "pip":
                run_pip(current_pip_args + [target])
            elif install_type == "git":
                 run_pip(current_pip_args + ['--upgrade', target]) # Use upgrade for git
            elif install_type == "wheel":
                wheel_path = Path(target)
                if not wheel_path.exists():
                    log_error(f"Required wheel file not found: {wheel_path}")
                    raise FileNotFoundError(f"Wheel not found: {wheel_path}")
                run_pip(current_pip_args + [str(wheel_path)])
            elif install_type == "wheel_url":
                 run_pip(current_pip_args + [target])
            elif install_type == "custom_copy":
                 repo_url = target
                 subdir_to_copy = name # Assumes 'name' is the directory name inside the repo

                 repo_hash = str(hash(repo_url))[:8]
                 clone_dir = user_data_path() / f"temp_clone_{repo_hash}"

                 # --- Cloning Logic ---
                 if repo_url not in cloned_repos:
                     # (Existing clone logic with removal)
                     log_info(f"Attempting to prepare clone directory: {clone_dir}")
                     if clone_dir.exists():
                         log_info("Removing existing temporary clone directory...")
                         shutil.rmtree(clone_dir, onerror=onerror_remove_readonly)
                         log_info("Successfully removed existing directory.")

                     log_info(f"Cloning {repo_url} to {clone_dir}...")
                     run_command(["git", "clone", "--recursive", "--depth", "1", repo_url, str(clone_dir)], error_message=f"Failed to clone {repo_url}")
                     cloned_repos[repo_url] = clone_dir
                 else:
                     clone_dir = cloned_repos[repo_url]
                     log_info(f"Using already cloned repo at {clone_dir} for {subdir_to_copy}")

                 # --- Copying Logic with Verification ---
                 source_subdir = clone_dir / subdir_to_copy
                 target_dir = site_packages_dir / subdir_to_copy

                 log_info(f"Verifying source for copy: {source_subdir}")
                 if not source_subdir.is_dir():
                     log_error(f"Subdirectory '{subdir_to_copy}' not found in cloned repo at {source_subdir}")
                     if clone_dir.exists():
                          log_info(f"Contents of {clone_dir}: {[p.name for p in clone_dir.iterdir()]}")
                     # Try alternative common repo structures
                     alt_source_subdir = clone_dir / "src" / subdir_to_copy
                     if alt_source_subdir.is_dir():
                          log_warning(f"Found source in 'src/' subdirectory: {alt_source_subdir}")
                          source_subdir = alt_source_subdir
                     else:
                          # One more try: maybe the repo *is* the package?
                          init_file = clone_dir / "__init__.py"
                          if init_file.exists() and subdir_to_copy == clone_dir.name:
                              log_warning(f"Repo root seems to be the package itself for '{subdir_to_copy}'")
                              source_subdir = clone_dir # Copy the whole clone dir
                          else:
                              raise FileNotFoundError(f"Source directory {source_subdir} (or variants) not found after cloning.")


                 log_info(f"Attempting to copy '{source_subdir}' to '{target_dir}'...")
                 if target_dir.exists():
                      log_info(f"Removing existing target directory: {target_dir}")
                      shutil.rmtree(target_dir, onerror=onerror_remove_readonly)

                 shutil.copytree(source_subdir, target_dir)
                 log_info(f"Copy successful: '{name}' should now be in site-packages.")

        except Exception as e:
            log_error(f"Failed during installation step for {name}: {e}")
            # (Existing cleanup logic for failed clone)
            if install_type == "custom_copy" and repo_url not in cloned_repos and clone_dir and clone_dir.exists():
                log_info(f"Cleaning up failed clone directory: {clone_dir}")
                shutil.rmtree(clone_dir, ignore_errors=True)
            raise

    # --- Final cleanup loop for temp clones ---
    log_info("Cleaning up temporary clone directories...")
    for repo_url, temp_clone_path in cloned_repos.items():
         if temp_clone_path.exists():
             try:
                 shutil.rmtree(temp_clone_path, onerror=onerror_remove_readonly)
                 log_info(f"Removed {temp_clone_path}")
             except Exception as e:
                 log_warning(f"Could not remove temporary clone directory {temp_clone_path}: {e}")

    log_info("Finished installing Trellis-specific dependencies.")


def install_trellis_dependencies(force_reinstall: bool = False):
    """Install specific dependencies needed by Trellis, often requiring wheels or git."""
    log_info("Installing Trellis-specific dependencies...")
    addon_root = addon_root_path()
    whl_dir = addon_root / "whl"

    # --- DEFINE BASE PIP FLAGS HERE TOO ---
    pip_flags = ['--no-cache-dir', '--no-warn-script-location', '--disable-pip-version-check']
    # --- END DEFINE BASE PIP FLAGS ---

    # Base args: install or install --force-reinstall
    pip_args_base = ['install']
    if force_reinstall:
        pip_args_base.append('--force-reinstall')

    # Define dependencies (same list as before)
    special_deps = [
        # Git Repos (install with dependencies)
        ("utils3d", "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8", "git", "Installing utils3d from git"),
        ("diffusers", "git+https://github.com/huggingface/diffusers.git", "git", "Installing diffusers from git"),
        ("transformers", "git+https://github.com/huggingface/transformers.git", "git", "Installing transformers from git"),
        ("accelerate", "git+https://github.com/huggingface/accelerate.git", "git", "Installing accelerate from git"),

        # Pip packages (install with dependencies)
        #("spconv", "spconv-cu118==2.3.6", "pip", "Installing spconv (cu118)"),
        ("spconv", "spconv-cu124", "pip", "Installing spconv (cu124)"),
        ("bitsandbytes", "bitsandbytes", "pip", "Installing bitsandbytes"),

        # Wheels (install WITHOUT dependencies by default, assuming torch/cuda handled)
        ("nvdiffrast", str(whl_dir / "nvdiffrast-0.3.3-cp311-cp311-win_amd64.whl"), "wheel", "Installing nvdiffrast from local wheel"),
        ("diffoctreerast", str(whl_dir / "diffoctreerast-0.0.0-cp311-cp311-win_amd64.whl"), "wheel", "Installing diffoctreerast from local wheel"),
        ("diff_gaussian_rasterization", str(whl_dir / "diff_gaussian_rasterization-0.0.0-cp311-cp311-win_amd64.whl"), "wheel", "Installing diff_gaussian_rasterization from local wheel"),
        ("flash_attn", "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.0.post2%2Bcu124torch2.4.1cxx11abiFALSE-cp311-cp311-win_amd64.whl", "wheel_url", "Installing flash_attn from URL (cu124)"),

        # Custom Copies (manual file copy, no pip involved for the package itself)
        ("trellis", "https://github.com/microsoft/TRELLIS.git", "custom_copy", "Copying Trellis library files"),
        ("api_spz", "https://github.com/IgorAherne/trellis-stable-projectorz.git", "custom_copy", "Copying api_spz library files"),
    ]

    site_packages_dir = get_site_packages_path()
    if not site_packages_dir:
         log_error("Cannot determine site-packages path. Aborting special dependency installation.")
         raise RuntimeError("Site-packages path not found.")

    cloned_repos = {}

    for name, target, install_type, comment in special_deps:
        log_info(comment)
        clone_dir = None
        # Start with base args (install or install --force-reinstall) + standard flags
        current_pip_args = pip_args_base + pip_flags

        needs_deps = install_type in ["git", "pip"]
        # No need to add/remove --no-deps explicitly here unless proven necessary
        # Let pip handle deps for git/pip installs, wheels are usually self-contained or rely on system/torch

        try:
            if install_type == "pip":
                run_pip(current_pip_args + [target])
            elif install_type == "git":
                 # Add --upgrade for git installs
                 run_pip(current_pip_args + ['--upgrade', target])
            elif install_type == "wheel":
                wheel_path = Path(target)
                if not wheel_path.exists():
                    log_error(f"Required wheel file not found: {wheel_path}")
                    raise FileNotFoundError(f"Wheel not found: {wheel_path}")
                run_pip(current_pip_args + [str(wheel_path)])
            elif install_type == "wheel_url":
                 run_pip(current_pip_args + [target])
            elif install_type == "custom_copy":
                 # ... (custom copy logic remains the same, including verification)
                 repo_url = target
                 subdir_to_copy = name # Assumes 'name' is the directory name inside the repo

                 repo_hash = str(hash(repo_url))[:8]
                 clone_dir = user_data_path() / f"temp_clone_{repo_hash}"

                 # --- Cloning Logic ---
                 if repo_url not in cloned_repos:
                     # (Existing clone logic with removal)
                     log_info(f"Attempting to prepare clone directory: {clone_dir}")
                     if clone_dir.exists():
                         log_info("Removing existing temporary clone directory...")
                         shutil.rmtree(clone_dir, onerror=onerror_remove_readonly)
                         log_info("Successfully removed existing directory.")

                     log_info(f"Cloning {repo_url} to {clone_dir}...")
                     run_command(["git", "clone", "--recursive", "--depth", "1", repo_url, str(clone_dir)], error_message=f"Failed to clone {repo_url}")
                     cloned_repos[repo_url] = clone_dir
                 else:
                     clone_dir = cloned_repos[repo_url]
                     log_info(f"Using already cloned repo at {clone_dir} for {subdir_to_copy}")

                 # --- Copying Logic with Verification ---
                 source_subdir = clone_dir / subdir_to_copy
                 target_dir = site_packages_dir / subdir_to_copy

                 log_info(f"Verifying source for copy: {source_subdir}")
                 if not source_subdir.is_dir():
                     log_error(f"Subdirectory '{subdir_to_copy}' not found in cloned repo at {source_subdir}")
                     if clone_dir.exists():
                          log_info(f"Contents of {clone_dir}: {[p.name for p in clone_dir.iterdir()]}")
                     alt_source_subdir = clone_dir / "src" / subdir_to_copy
                     if alt_source_subdir.is_dir():
                          log_warning(f"Found source in 'src/' subdirectory: {alt_source_subdir}")
                          source_subdir = alt_source_subdir
                     else:
                          init_file = clone_dir / "__init__.py"
                          if init_file.exists() and subdir_to_copy == clone_dir.name:
                              log_warning(f"Repo root seems to be the package itself for '{subdir_to_copy}'")
                              source_subdir = clone_dir
                          else:
                              raise FileNotFoundError(f"Source directory {source_subdir} (or variants) not found after cloning.")


                 log_info(f"Attempting to copy '{source_subdir}' to '{target_dir}'...")
                 if target_dir.exists():
                      log_info(f"Removing existing target directory: {target_dir}")
                      shutil.rmtree(target_dir, onerror=onerror_remove_readonly)

                 shutil.copytree(source_subdir, target_dir)
                 log_info(f"Copy successful: '{name}' should now be in site-packages.")


        except Exception as e:
            log_error(f"Failed during installation step for {name}: {e}")
            # (Existing cleanup logic for failed clone)
            if install_type == "custom_copy" and repo_url not in cloned_repos and clone_dir and clone_dir.exists():
                log_info(f"Cleaning up failed clone directory: {clone_dir}")
                shutil.rmtree(clone_dir, ignore_errors=True)
            raise

    log_info("Cleaning up temporary clone directories...")
    for repo_url, temp_clone_path in cloned_repos.items():
         if temp_clone_path.exists():
             try:
                 shutil.rmtree(temp_clone_path, onerror=onerror_remove_readonly)
                 log_info(f"Removed {temp_clone_path}")
             except Exception as e:
                 log_warning(f"Could not remove temporary clone directory {temp_clone_path}: {e}")

    log_info("Finished installing Trellis-specific dependencies.")

#    except NameError as ne: # Catch the specific error if it somehow still happens
#        log_error(f"A NameError occurred during installation: {ne}. This likely indicates a coding error.")
#        if prefs: prefs.dependencies_installed = False
#        raise ne # Re-raise NameError to make it obvious


def parse_package_name(package_line):
    """Parse package name from requirements.txt line (handles versions, extras)."""
    package_line = package_line.strip()
    if not package_line or package_line.startswith('#'):
        return None
    # Remove comments
    package_line = package_line.split('#')[0].strip()
    # Basic split for version specifiers, git urls etc.
    # Handles git+, http://, package_name==, package_name>= etc.
    match = re.match(r"^(?:git\+|https?:\/\/.*#egg=)?([a-zA-Z0-9_\-\.]+)", package_line)
    if match:
        name = match.group(1).replace('-', '_').split('[')[0]
        return name
    # Fallback for simple names without specifiers
    match_simple = re.match(r"([a-zA-Z0-9_\-\.]+)", package_line)
    if match_simple:
        name = match_simple.group(1).replace('-', '_').split('[')[0]
        return name

    log_warning(f"Could not parse package name from line: {package_line}")
    return None


# --- UPDATED check_dependencies function ---
def check_dependencies() -> bool:
    """Check if all required packages (requirements + special) are importable."""
    log_info("Checking dependencies...")
    requirements_txt = addon_root_path() / "requirements.txt"
    missing_packages = []
    packages_checked = set() # Track packages already checked

    if not venv_path().exists():
        log_warning("Virtual environment directory not found.")
        return False # Cannot check if venv doesn't exist

    # Ensure venv is in path for this check
    add_venv_to_sys_path()

    # Define expected import names and map package names to import names
    # This map helps handle cases like 'Pillow' -> 'PIL' or 'scikit-image' -> 'skimage'
    # It also includes packages installed specially.
    package_to_import_map = {
        # From requirements (examples, add more if needed)
        "Pillow": "PIL",
        "scikit_image": "skimage",
        "onnxruntime_gpu": "onnxruntime", # Assuming GPU version installs 'onnxruntime' import
        "opencv_python": "cv2",
        "opencv_python_headless": "cv2",
        "invisible_watermark": "imWatermark", # Check the actual import name
        "pytorch": "torch", # Alias for consistency if 'pytorch' is in reqs
        "torch": "torch", # Base torch
        "torchvision": "torchvision",
        "torchaudio": "torchaudio",
        "xformers": "xformers",
        # From special_deps
        "utils3d": "utils3d",
        #"spconv_cu118": "spconv", # Package name maps to import name
        "spconv_cu124": "spconv", # Handle potential variations
        "spconv": "spconv",       # Direct name
        "diffusers": "diffusers",
        "transformers": "transformers", # Diffusers needs this
        "nvdiffrast": "nvdiffrast",
        "diffoctreerast": "diffoctreerast",
        "diff_gaussian_rasterization": "diff_gaussian_rasterization",
        "flash_attn": "flash_attn",
        "accelerate": "accelerate",
        "trellis": "trellis", # From custom_copy
        "api_spz": "api_spz", # From custom_copy
        "huggingface_hub": "huggingface_hub",
        "numpy": "numpy",
        "bitsandbytes": "bitsandbytes", # For quantization
        # Add any other direct package names from requirements if they differ from import name
    }

    # --- List of packages/imports to attempt importing ---
    # Start with essentials and special ones
    packages_to_check: Set[str] = {
        "torch", "torchvision", "torchaudio", "xformers", # PyTorch stack
        "diffusers", "transformers", "accelerate", # Hugging Face stack
        "PIL", # Pillow
        "skimage", # scikit-image
        "cv2", # OpenCV
        "onnxruntime", # ONNX Runtime
        "huggingface_hub",
        "numpy",
        "scipy", # Often needed by image/scientific libs
        "bitsandbytes", # If used for quantization
        "imWatermark", # Invisible watermark
        # Trellis specific (installed specially)
        "utils3d",
        "spconv",
        "nvdiffrast",
        "diffoctreerast",
        "diff_gaussian_rasterization",
        "flash_attn",
        "trellis", # From custom copy
        "api_spz", # From custom copy
    }

    # --- Add packages parsed from requirements.txt ---
    if requirements_txt.exists():
        try:
            with open(requirements_txt, 'r') as f:
                for line in f:
                    package_name_parsed = parse_package_name(line)
                    if package_name_parsed:
                        # Get the corresponding import name from the map, or use the parsed name directly
                        import_name = package_to_import_map.get(package_name_parsed, package_name_parsed)
                        packages_to_check.add(import_name)
                        # Handle cases where one package provides multiple importable modules (like torch)
                        if import_name == "torch":
                             packages_to_check.update(["torchvision", "torchaudio"])

        except FileNotFoundError:
            log_error(f"Requirements file not found at {requirements_txt} during check.")
            # Don't necessarily fail here, maybe special deps are enough
        except Exception as e:
            log_error(f"Error reading requirements file during check: {e}")
            return False # Fail if we can't read the requirements file properly


    # --- Perform the import checks ---
    log_info(f"Attempting to import: {sorted(list(packages_to_check))}")
    for import_name in sorted(list(packages_to_check)): # Check in alphabetical order for logging
        if import_name in packages_checked:
            continue # Skip if already checked (e.g., added explicitly and also from reqs)

        try:
            # Special case: Check CUDA availability for torch if 'cuda' device intended
            if import_name == "torch":
                 module = importlib.import_module(import_name)
                 debug_print(f"Package '{import_name}' found.")
                 # Optional: Add a specific check for CUDA if GPU is expected
                 # if 'cuda' in get_torch_device(): # Careful: get_torch_device might try importing torch itself
                 #     if not module.cuda.is_available():
                 #         log_warning("PyTorch imported, but CUDA is not available!")
                 #         # Don't mark as missing, but log warning. Decide if this should be an error.
            else:
                 importlib.import_module(import_name)
                 debug_print(f"Package '{import_name}' found.")

            packages_checked.add(import_name)

        except ImportError:
            log_warning(f"Required package '{import_name}' is missing or not importable.")
            missing_packages.append(import_name)
            packages_checked.add(import_name) # Mark as checked even if missing
        except ModuleNotFoundError: # More specific subclass of ImportError
            log_warning(f"Required package '{import_name}' not found (ModuleNotFoundError).")
            missing_packages.append(import_name)
            packages_checked.add(import_name)
        except Exception as e:
             # Catch broader errors (like the DLL load error for flash_attn)
             log_error(f"Error importing package '{import_name}': {e}")
             # Include traceback for detailed debugging if needed
             # import traceback
             # log_error(traceback.format_exc())
             missing_packages.append(f"{import_name} (import error: {type(e).__name__})") # Add error type to missing list
             packages_checked.add(import_name) # Mark as checked even if error occurred

    # --- Final Result ---
    if not missing_packages:
        log_info("All required dependencies seem to be installed and importable.")
        return True # Success
    else:
        log_error(f"Missing or non-importable packages/modules: {', '.join(sorted(list(set(missing_packages))))}")
        return False # Failure


def uninstall_packages_and_venv():
    """Uninstall packages and remove the virtual environment."""
    log_info("Starting uninstallation process...")
    venv_p = venv_path()

    if not venv_p.exists():
        log_info("Virtual environment directory not found. Nothing to uninstall.")
        return

    # Remove the entire venv directory robustly
    try:
        log_info(f"Removing virtual environment directory: {venv_p}")
        shutil.rmtree(venv_p, onerror=onerror_remove_readonly)
        log_info("Virtual environment removed successfully.")
        if ADDON_ID in bpy.context.preferences.addons:
             prefs = get_addon_pref()
             if prefs:
                 prefs.dependencies_installed = False

    except Exception as e:
        log_error(f"Failed to remove virtual environment directory: {e}")
        log_error(f"Please manually delete the directory: {venv_p}")


# --- Addon Preferences --- (Keep existing Preferences class)
class AssetGeneratorPreferences(AddonPreferences):
    bl_idname = ADDON_ID

    dependencies_installed: BoolProperty(
        name="Dependencies Installed Flag",
        description="Indicates if dependencies have been installed (best effort check)",
        default=False
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="Dependency Management:")

        venv_p = venv_path()
        if venv_p.exists():
            layout.label(text=f"Virtual Env: {venv_p}", icon='FOLDER_REDIRECT')
            if self.dependencies_installed:
                 box = layout.box()
                 box.label(text="Dependencies seem to be installed.", icon='CHECKMARK')
                 row = box.row()
                 op_reinstall = row.operator(InstallDependenciesOperator.bl_idname, text="Reinstall Dependencies", icon='FILE_REFRESH')
                 op_reinstall.force_reinstall = True # Set flag for the operator instance
                 row.operator(CheckDependenciesOperator.bl_idname, text="Verify", icon='VIEWZOOM')

            else:
                 box = layout.box()
                 box.label(text="Dependencies may be missing or incomplete.", icon='ERROR')
                 row = box.row()
                 op_install = row.operator(InstallDependenciesOperator.bl_idname, text="Install Dependencies", icon='IMPORT')
                 op_install.force_reinstall = False
                 row.operator(CheckDependenciesOperator.bl_idname, text="Check Again", icon='VIEWZOOM')

            layout.operator(UninstallDependenciesOperator.bl_idname, text="Uninstall All & Remove Env", icon='TRASH')

        else:
            layout.label(text="Virtual environment not found.", icon='QUESTION')
            op_install = layout.operator(InstallDependenciesOperator.bl_idname, text="Install Dependencies", icon='IMPORT')
            op_install.force_reinstall = False

        layout.separator()

# --- Operators for Preferences --- (Keep existing Operators: Install, Check, Uninstall)
# Small tweak to InstallDependenciesOperator to reset force_reinstall if needed
class InstallDependenciesOperator(bpy.types.Operator):
    bl_idname = "asset_generator.install_dependencies"
    bl_label = "Install/Reinstall Dependencies"
    bl_description = "Creates a virtual environment (if needed) and installs required Python packages. May take a long time."

    force_reinstall: BoolProperty(
        name="Force Reinstall",
        description="Forces reinstallation of all packages, ignoring cache and existing installs.",
        default=False
    )

    @classmethod
    def description(cls, context, properties):
         if properties.force_reinstall:
             return "Downloads and reinstalls ALL packages, ignoring existing ones. Slower, use if installs seem broken."
         else:
             return "Downloads and installs required Python packages, upgrading existing ones. Standard install."


    def execute(self, context):
        prefs = get_addon_pref()
        try:
            action = "Reinstalling" if self.force_reinstall else "Installing"
            self.report({'INFO'}, f"Starting dependency {action.lower()}... This may take several minutes.")
            # Ensure Blender UI updates if possible (might still freeze during subprocess calls)
            bpy.context.window_manager.progress_begin(0, 1)
            bpy.context.window_manager.progress_update(0)
            # Force UI update
            # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)


            install_packages(force_reinstall=self.force_reinstall)
            prefs.dependencies_installed = check_dependencies() # Verify after installation

            bpy.context.window_manager.progress_end()

            if prefs.dependencies_installed:
                self.report({'INFO'}, f"Dependencies {action.lower()}ed successfully.")
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, f"Dependency {action.lower()} finished, but some packages might be missing or failed to import. Check logs.")
                return {'CANCELLED'}

        except Exception as e:
            log_error(f"{action} failed: {e}")
            if prefs: # Ensure prefs exists before setting attr
                prefs.dependencies_installed = False
            bpy.context.window_manager.progress_end() # Ensure progress ends on error
            self.report({'ERROR'}, f"{action} failed. Check Blender console for details: {e}")
            return {'CANCELLED'}
        # finally:
            # Reset force_reinstall for next time operator is called without explicit button press?
            # self.force_reinstall = False # Probably not needed as buttons set it explicitly

class CheckDependenciesOperator(bpy.types.Operator):
    bl_idname = "asset_generator.check_dependencies"
    bl_label = "Check Dependencies"
    bl_description = "Verify if required Python packages are installed and importable"

    def execute(self, context):
        prefs = get_addon_pref()
        if not prefs:
             self.report({'ERROR'}, "Could not access addon preferences.")
             return {'CANCELLED'}
        try:
            # Ensure Blender UI updates before the potentially long check
            # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            bpy.context.window_manager.progress_begin(0, 1) # Show progress for check too
            bpy.context.window_manager.progress_update(0)

            check_successful = check_dependencies() # Run the updated check
            prefs.dependencies_installed = check_successful

            bpy.context.window_manager.progress_end()

            if check_successful:
                 self.report({'INFO'}, "Dependency check successful. All required packages seem importable.")
            else:
                 self.report({'WARNING'}, "Dependency check failed or found missing/broken packages. See console for details.")

        except Exception as e:
            if prefs: prefs.dependencies_installed = False
            bpy.context.window_manager.progress_end()
            self.report({'ERROR'}, f"Error during dependency check: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}

class UninstallDependenciesOperator(bpy.types.Operator):
    bl_idname = "asset_generator.uninstall_dependencies"
    bl_label = "Uninstall Dependencies & Env"
    bl_description = "Remove all installed packages and the virtual environment directory"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        prefs = get_addon_pref()
        try:
            bpy.context.window_manager.progress_begin(0, 1) # Show progress
            bpy.context.window_manager.progress_update(0)
            uninstall_packages_and_venv()
            if prefs:
                prefs.dependencies_installed = False # Reset flag
            bpy.context.window_manager.progress_end()
            self.report({'INFO'}, "Dependencies and virtual environment uninstalled.")
        except Exception as e:
            bpy.context.window_manager.progress_end()
            self.report({'ERROR'}, f"Error during uninstallation: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}


# --- Core Functionality --- (Keep existing core functions: get_torch_device, flush_gpu_memory, etc.)
# Assume the VenvImportGuard is used correctly before imports in these core functions
def get_torch_device() -> str:
    """Determine the best available torch device."""
    # Cache the device to avoid repeated checks
    if not hasattr(get_torch_device, "device"):
        try:
            # Ensure venv is active for import
            add_venv_to_sys_path()
            import torch
            if torch.cuda.is_available():
                get_torch_device.device = "cuda"
            # elif torch.backends.mps.is_available() and torch.backends.mps.is_built(): # MPS Check
            #      get_torch_device.device = "mps"
            else:
                get_torch_device.device = "cpu"
            log_info(f"Using torch device: {get_torch_device.device}")
        except ImportError:
            log_warning("Torch not importable during device check. Dependencies might be missing. Defaulting to CPU.")
            get_torch_device.device = "cpu"
        except Exception as e:
             log_error(f"Error detecting torch device: {e}. Defaulting to CPU.")
             get_torch_device.device = "cpu"
    return get_torch_device.device

def flush_gpu_memory():
    """Clear PyTorch CUDA cache and run Python garbage collection."""
    device = get_torch_device()
    log_info("Flushing memory...")
    gc.collect() # Python garbage collection
    if device == "cuda":
        try:
            import torch
            torch.cuda.empty_cache()
            log_info("CUDA cache flushed.")
        except ImportError:
            debug_print("Torch not found, cannot flush CUDA cache.")
        except Exception as e:
            log_error(f"Error flushing CUDA cache: {e}")


# Get a list of text blocks in Blender
def get_text_blocks(self, context):
    """EnumProperty callback: Get available text blocks."""
    items = [(text.name, text.name, "") for text in bpy.data.texts]
    if not items:
        items.append(("NONE", "No Text Blocks Found", ""))
    return items


# Property Group for storing the selected text block and toggle
class AssetGenInputProps(PropertyGroup):

    def update_text_selection(self, context):
        """Update helper for text block selection (optional)."""
        # Example: could potentially load text content here if needed elsewhere
        pass

    input_type: EnumProperty(
        name="Input Source",
        description="Choose the source for the generation prompt",
        items=[
            ("PROMPT", "Prompt", "Use the text prompt input below"),
            ("TEXT_BLOCK", "Text Block", "Use lines from a selected Text Editor block"),
        ],
        default="PROMPT",
    )

    # Use StringProperty for prompt, simplifies UI connection
    # asset_prompt: StringProperty(...) defined on Scene instead

    # Enum for selecting the text block
    text_block_name: EnumProperty(
        name="Text Block",
        items=get_text_blocks,
        update=update_text_selection,
        description="Select the Text Block to use as input (one asset per line)",
    )


def sanitize_filename(name: str) -> str:
    """Remove characters unsafe for filenames."""
    # Remove or replace problematic characters: <>:"/\|?* and potentially others
    # Also remove leading/trailing whitespace
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'[\x00-\x1f\x7f]', '_', name) # Control characters
    name = name.strip()
    if not name: # Handle empty names
         name = "Unnamed Asset"
    return name


def get_unique_object_name(base_name: str) -> str:
    """Generates a unique Blender object name (e.g., "Cube", "Cube.001")."""
    if base_name not in bpy.data.objects:
        return base_name

    # Use Blender's built-in naming convention check
    i = 1
    while True:
        new_name = f"{base_name}.{i:03d}"
        if new_name not in bpy.data.objects:
            return new_name
        i += 1


def get_unique_path(path_str: str) -> Path:
    """Generates a unique file path if the given path exists."""
    path = Path(path_str)
    if not path.exists():
        return path

    base = path.stem
    ext = path.suffix
    parent = path.parent

    # Simple counter suffix - adjust regex if Blender-style (.001) is preferred
    counter = 1
    # Remove existing counter suffix if present (e.g., "file (1).png")
    match = re.search(r"_\((\d+)\)$", base)
    if match:
         base = base[:match.start()]
         counter = int(match.group(1)) + 1

    while True:
        # Use underscore before parentheses for clarity
        new_name = f"{base}_({counter}){ext}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1


# --- Main Generation Operator ---
class ASSET_GENERATOR_OT_Generate(bpy.types.Operator):
    """Generate asset image(s), convert to 3D plane(s) and GLB model(s)"""

    bl_idname = "asset_generator.generate"
    bl_label = "Generate Asset(s)"
    bl_options = {"REGISTER", "UNDO"}

    # --- Internal State & Models (cached) ---
    _flux_pipe = None
    _birefnet_model = None
    _trellis_pipe = None

    # --- Helper Methods ---
#    def _ensure_dependencies(self) -> bool:
#        """Check if dependencies are installed before running."""
#        prefs = get_addon_pref()
#        if not prefs.dependencies_installed:
#            # Run a quick check just in case
#            if not check_dependencies():
#                self.report({'ERROR'}, "Dependencies not installed. Please install via Add-on Preferences.")
#                return False
#            else:
#                prefs.dependencies_installed = True # Update flag if check passes
#                return True
#        return True

    def _get_flux_pipeline(self):
        """Load or return cached FLUX pipeline."""
        if self._flux_pipe is None:
            log_info("Loading FLUX pipeline...")
            try:
                add_venv_to_sys_path() # Ensure import works
                from diffusers import FluxPipeline, BitsAndBytesConfig, FluxTransformer2DModel
                import torch

                # Configuration (Adjust model card/quantization as needed)
                # image_model_card = "black-forest-labs/FLUX.1-schnell" # Faster, less VRAM
                image_model_card = "black-forest-labs/FLUX.1-dev" # Slower, potentially better quality

                # Optional: 4-bit quantization (requires bitsandbytes)
                use_quantization = True
                if use_quantization:
                    try:
                        nf4_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16, # Or float16 if bfloat16 not supported
                        )
                        model_nf4 = FluxTransformer2DModel.from_pretrained(
                            image_model_card,
                            subfolder="transformer",
                            quantization_config=nf4_config,
                            torch_dtype=torch.bfloat16,
                        )
                        self._flux_pipe = FluxPipeline.from_pretrained(
                             image_model_card,
                             transformer=model_nf4,
                             torch_dtype=torch.bfloat16
                        )
                        log_info("Loaded FLUX with 4-bit quantization.")
                    except Exception as e:
                         log_warning(f"Failed to load FLUX with 4-bit quantization ({e}). Falling back to default.")
                         self._flux_pipe = FluxPipeline.from_pretrained(image_model_card, torch_dtype=torch.bfloat16)

                else:
                    self._flux_pipe = FluxPipeline.from_pretrained(image_model_card, torch_dtype=torch.bfloat16)


                # Move to device and enable optimizations
                device = get_torch_device()
                if device == "cpu":
                     log_warning("Running FLUX on CPU. This will be very slow.")
                     # No specific CPU offload needed for CPU execution
                elif device == "mps":
                     log_info("Moving FLUX pipeline to MPS device.")
                     self._flux_pipe.to("mps")
                     # Potentially add MPS specific optimizations if needed/available
                else: # cuda
                     log_info("Moving FLUX pipeline to CUDA device and enabling CPU offload.")
                     # Enable CPU offloading for potentially lower VRAM usage
                     self._flux_pipe.enable_model_cpu_offload()

                # VAE optimizations (can help with memory)
                # self._flux_pipe.enable_vae_slicing()
                # self._flux_pipe.vae.enable_tiling()

            except ImportError as e:
                log_error(f"Failed to import diffusers/torch: {e}. Dependencies missing?")
                raise
            except Exception as e:
                log_error(f"Failed to load FLUX pipeline: {e}")
                raise # Critical error

        return self._flux_pipe

    def _get_birefnet_model(self):
        """Load or return cached BiRefNet model for background removal."""
        if self._birefnet_model is None:
            log_info("Loading BiRefNet segmentation model...")
            try:
                add_venv_to_sys_path()
                from transformers import AutoModelForImageSegmentation
                import torch

                model_name = "ZhengPeng7/BiRefNet"
                self._birefnet_model = AutoModelForImageSegmentation.from_pretrained(model_name, trust_remote_code=True)

                device = get_torch_device()
                self._birefnet_model.to(device)
                self._birefnet_model.eval() # Set to evaluation mode
                log_info(f"BiRefNet model loaded to {device}.")
            except ImportError as e:
                log_error(f"Failed to import transformers/torch: {e}. Dependencies missing?")
                raise
            except Exception as e:
                log_error(f"Failed to load BiRefNet model: {e}")
                raise # Critical error
        return self._birefnet_model


    def _get_trellis_pipeline(self):
        """Load or return cached Trellis pipeline."""
        if self._trellis_pipe is None:
            log_info("Loading Trellis Image-to-3D pipeline...")
            try:
                add_venv_to_sys_path()
                # Set environment variable needed by spconv BEFORE importing Trellis/torch
                os.environ['SPCONV_ALGO'] = 'native' # Or try other values like 'implicit_gemm' if needed
                debug_print(f"Set SPCONV_ALGO='{os.environ['SPCONV_ALGO']}'")

                from trellis.pipelines import TrellisImageTo3DPipeline
                import torch

                # Ensure torch is available and device detected before loading
                device = get_torch_device()
                if device == "cpu":
                     log_warning("Trellis pipeline requires a CUDA GPU. Running on CPU is not supported/tested.")
                     # Raise an error or return None to skip GLB generation?
                     # raise RuntimeError("Trellis pipeline requires CUDA.")
                     return None # Skip GLB generation if no CUDA

                # Load the pipeline (adjust model name if needed)
                model_name = "JeffreyXiang/TRELLIS-image-large"
                self._trellis_pipe = TrellisImageTo3DPipeline.from_pretrained(model_name)

                # Move to CUDA device
                self._trellis_pipe.to("cuda") # Trellis likely expects CUDA specifically
                log_info("Trellis pipeline loaded to CUDA device.")

            except ImportError as e:
                log_error(f"Failed to import trellis/torch: {e}. Dependencies missing?")
                raise
            except Exception as e:
                log_error(f"Failed to load Trellis pipeline: {e}")
                # Don't raise critical error, allow fallback without GLB
                return None # Indicate failure to load

        return self._trellis_pipe


    def _generate_raw_image(self, prompt: str, output_dir: Path, base_filename: str) -> Path:
        """Generate image using FLUX, save it."""
        flux_pipe = self._get_flux_pipeline()
        log_info(f"Generating image for prompt: '{prompt}'")

        # Ensure output path is unique BEFORE generation
        raw_image_path = get_unique_path(str(output_dir / f"{base_filename}_raw.png"))

        try:
            # Parameters (tune as needed)
            full_prompt = "neutral background, " + prompt # Prepend background hint
            guidance_scale = 3.0 # Lower values for more prompt adherence
            num_inference_steps = 25 # Faster generation
            height=1024
            width=1024
            # max_sequence_length=256 # Only needed for FLUX dev usually

            # Generate
            with VenvImportGuard(): # Ensure imports work if called standalone
                 import torch
                 with torch.no_grad(): # Conserve memory
                     image = flux_pipe(
                         prompt=full_prompt,
                         guidance_scale=guidance_scale,
                         height=height,
                         width=width,
                         num_inference_steps=num_inference_steps,
                         # max_sequence_length=max_sequence_length, # If using FLUX dev
                     ).images[0]

            # Save
            image.save(raw_image_path)
            log_info(f"Raw image saved to: {raw_image_path}")
            return raw_image_path

        except Exception as e:
            log_error(f"Image generation failed: {e}")
            flush_gpu_memory() # Attempt to clear memory on failure
            raise # Re-raise

    def _remove_background(self, input_image_path: Path, output_dir: Path, base_filename: str) -> Path:
        """Remove background using BiRefNet, save transparent image."""
        birefnet_model = self._get_birefnet_model()
        log_info(f"Removing background from: {input_image_path}")

        transparent_image_path = get_unique_path(str(output_dir / f"{base_filename}_transparent.png"))

        try:
             with VenvImportGuard():
                 from PIL import Image
                 from torchvision import transforms
                 import torch

                 # Load image
                 image = Image.open(input_image_path).convert("RGB")
                 original_size = image.size

                 # Prepare input tensor
                 # Match BiRefNet expected normalization
                 transform = transforms.Compose([
                     transforms.Resize((1024, 1024)), # Model expects this size
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 ])
                 input_tensor = transform(image).unsqueeze(0).to(get_torch_device())

                 # Predict mask
                 with torch.no_grad():
                     # Model output structure might vary, check documentation or debug output
                     # Assuming last output is the primary mask prediction
                     outputs = birefnet_model(input_tensor)
                     # Accessing the logits, often the first element or a specific key
                     logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
                     # Apply sigmoid and get mask
                     preds = logits.sigmoid().cpu() # Move to CPU for PIL conversion

                 # Process mask
                 pred_mask = preds[0].squeeze() # Remove batch and channel dims
                 mask_image = transforms.ToPILImage()(pred_mask)
                 mask_image = mask_image.resize(original_size, Image.LANCZOS) # Resize back smoothly

                 # Refine mask (optional but recommended)
                 refined_mask = self._refine_mask(mask_image, threshold=128, feather_radius=2)

                 # Apply mask
                 image.putalpha(refined_mask)

                 # Save
                 image.save(transparent_image_path)
                 log_info(f"Transparent image saved to: {transparent_image_path}")
                 return transparent_image_path

        except Exception as e:
            log_error(f"Background removal failed: {e}")
            flush_gpu_memory()
            raise

    def _refine_mask(self, mask: "Image", threshold=128, feather_radius=1) -> "Image":
        """Apply thresholding and feathering to a PIL mask image."""
        with VenvImportGuard():
             from PIL import ImageFilter

             mask = mask.convert("L")
             # Thresholding: Convert to binary mask
             mask = mask.point(lambda p: 255 if p > threshold else 0, mode='L')
             # Feathering: Apply Gaussian blur to soften edges
             if feather_radius > 0:
                 mask = mask.filter(ImageFilter.GaussianBlur(feather_radius))
             return mask

    def _crop_to_content(self, image_path: Path) -> Path:
         """Crop image to the bounding box of non-transparent pixels. Overwrites original."""
         try:
             with VenvImportGuard():
                 from PIL import Image
                 img = Image.open(image_path).convert("RGBA")
                 bbox = img.getbbox() # Gets bounding box of non-zero alpha
                 if bbox:
                     cropped_img = img.crop(bbox)
                     cropped_img.save(image_path) # Overwrite with cropped version
                     log_info(f"Image cropped to content: {image_path}")
                 else:
                     log_warning(f"Image has no content to crop: {image_path}")
             return image_path
         except Exception as e:
             log_error(f"Failed to crop image {image_path}: {e}")
             return image_path # Return original path on failure


    def _split_alpha_islands(self, image_path: Path, output_dir: Path, base_filename: str) -> List[Path]:
        """Split image into parts based on disconnected alpha components."""
        log_info(f"Splitting image by alpha islands: {image_path}")
        try:
            with VenvImportGuard():
                from PIL import Image
                import numpy as np
                from scipy.ndimage import label, find_objects

                img = Image.open(image_path).convert("RGBA")
                img_data = np.array(img)
                alpha_mask = img_data[:, :, 3] > 10 # Use a small threshold for alpha mask

                labeled_array, num_features = label(alpha_mask)
                log_info(f"Found {num_features} alpha islands.")

                if num_features <= 1:
                     log_info("Image has 0 or 1 island, no splitting needed.")
                     # Return the original path in a list if it has content
                     if np.any(alpha_mask):
                           return [image_path]
                     else:
                           log_warning("Image is fully transparent, skipping.")
                           return []


                saved_paths = []
                bboxes = find_objects(labeled_array) # Get bounding boxes for each label

                for i, bbox in enumerate(bboxes, start=1):
                    if bbox is None: continue # Should not happen if num_features > 0

                    # Extract the component using the bounding box
                    # Slicing is [y_slice, x_slice]
                    y_slice, x_slice = bbox
                    component_mask = (labeled_array[y_slice, x_slice] == i)
                    component_data = img_data[y_slice, x_slice]

                    # Apply the component mask to make background transparent
                    component_data[~component_mask] = [0, 0, 0, 0]

                    # Create PIL image from the component data
                    character_img = Image.fromarray(component_data, 'RGBA')

                    # Crop again to the actual content of the island
                    char_bbox = character_img.getbbox()
                    if char_bbox:
                         character_img = character_img.crop(char_bbox)
                    else:
                         log_warning(f"Skipping empty island {i}")
                         continue # Skip empty islands


                    # Save each cropped component
                    part_path = get_unique_path(str(output_dir / f"{base_filename}_part_{i}.png"))
                    character_img.save(part_path)
                    saved_paths.append(part_path)
                    debug_print(f"Saved asset part: {part_path}")

                return saved_paths

        except ImportError as e:
            log_error(f"Failed to import numpy/scipy: {e}. Splitting requires these dependencies.")
            # Fallback: return the original path if imports fail
            return [image_path]
        except Exception as e:
            log_error(f"Failed to split image islands: {e}")
            # Fallback: return the original path
            return [image_path]


    def _create_blender_plane(self, image_path: Path, asset_name: str, prompt: str) -> Optional[bpy.types.Object]:
        """Create a textured plane object in Blender from the image."""
        log_info(f"Creating Blender plane for: {image_path}")
        if not image_path.exists():
            log_error(f"Image not found: {image_path}")
            return None

        try:
             with VenvImportGuard():
                 from PIL import Image # Needed to get image dimensions

             img_pil = Image.open(image_path)
             img_width, img_height = img_pil.size
             aspect_ratio = img_width / img_height if img_height > 0 else 1.0
             img_pil.close() # Close the image file

             # --- Create Material ---
             mat_name = get_unique_object_name(f"{asset_name}_Mat") # Use object name logic for materials too
             material = bpy.data.materials.new(name=mat_name)
             material.use_nodes = True
             nodes = material.node_tree.nodes
             links = material.node_tree.links

             # Clear default nodes
             nodes.clear()

             # Create necessary nodes
             tex_image = nodes.new(type='ShaderNodeTexImage')
             bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
             output = nodes.new(type='ShaderNodeOutputMaterial')

             # Load image texture
             tex_image.image = bpy.data.images.load(str(image_path), check_existing=True)
             tex_image.interpolation = 'Linear'
             tex_image.location = (-300, 300)

             # Configure BSDF
             bsdf.location = (0, 300)
             # Connect Base Color and Alpha
             links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
             links.new(bsdf.inputs['Alpha'], tex_image.outputs['Alpha'])
             # Set other BSDF parameters for typical image planes if needed (e.g., roughness)
             bsdf.inputs['Roughness'].default_value = 0.8
             bsdf.inputs['Specular'].default_value = 0.1

             # Configure Output
             output.location = (300, 300)
             links.new(output.inputs['Surface'], bsdf.outputs['BSDF'])

             # Set material settings for transparency
             material.blend_method = 'CLIP' # Or 'HASHED'/'BLEND'
             material.shadow_method = 'CLIP' # Or 'HASHED'/'OPAQUE'
             # material.alpha_threshold = 0.5 # Adjust threshold for CLIP mode

             # --- Create Plane Object ---
             bpy.ops.mesh.primitive_plane_add(size=1.0, enter_editmode=False, align='WORLD', location=bpy.context.scene.cursor.location)
             obj = bpy.context.active_object
             obj.name = get_unique_object_name(asset_name) # Ensure unique object name

             # Assign material
             obj.data.materials.append(material)

             # Adjust plane size and orientation
             # Scale based on aspect ratio (keep Y=1, scale X)
             obj.scale = (aspect_ratio, 1.0, 1.0)
             # Rotate to stand upright (assuming Z is up)
             obj.rotation_euler = (math.radians(90), 0, 0)
             bpy.ops.object.transform_apply(location=False, rotation=True, scale=True) # Apply rotation & scale

             # Optional: Move origin to bottom center
             bpy.ops.object.mode_set(mode='EDIT')
             bpy.ops.mesh.select_all(action='SELECT')
             bpy.ops.transform.translate(value=(0, 0, 0.5)) # Move geometry up by half height
             bpy.ops.object.mode_set(mode='OBJECT')
             bpy.ops.object.transform_apply(location=False, rotation=False, scale=False, properties=True) # Apply mesh changes
             bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN') # Set origin to cursor (at original location)

             # --- Mark as Asset ---
             obj.asset_mark()
             obj.asset_data.author = bl_info["author"]
             obj.asset_data.description = prompt
             # Add tags
             tag_list = ["Generated", "2D Asset", "Plane"]
             base_prompt_words = re.findall(r'\b\w+\b', prompt.lower())[:3] # Add first few words of prompt as tags
             tag_list.extend([word.capitalize() for word in base_prompt_words])
             for tag_name in tag_list:
                  obj.asset_data.tags.new(name=tag_name.strip())

             # Generate asset preview (use the same image)
             try:
                 with context.temp_override(id=obj): # Override context for preview generation
                     bpy.ops.ed.lib_id_load_custom_preview(filepath=str(image_path))
             except Exception as e:
                  log_warning(f"Could not generate asset preview for {obj.name}: {e}")

             log_info(f"Created Blender asset: {obj.name}")
             return obj

        except Exception as e:
            log_error(f"Failed to create Blender plane for {image_path}: {e}")
            # Clean up partially created object/material?
            return None


    def _generate_glb(self, image_path: Path, output_dir: Path, base_filename: str) -> Optional[Path]:
        """Generate GLB model from image using Trellis."""
        trellis_pipe = self._get_trellis_pipeline()
        if trellis_pipe is None:
            log_warning("Trellis pipeline not available. Skipping GLB generation.")
            return None

        log_info(f"Generating GLB for: {image_path}")
        glb_path = get_unique_path(str(output_dir / f"{base_filename}.glb"))

        try:
            with VenvImportGuard():
                from PIL import Image
                from trellis.utils import postprocessing_utils # Keep render_utils if needed later
                import torch

                # Load image
                input_image = Image.open(image_path)

                # Run Trellis pipeline
                with torch.no_grad():
                    outputs = trellis_pipe.run(input_image, seed=None) # Use random seed

                # Postprocess to GLB
                # Adjust simplification and texture size as needed
                glb_data = postprocessing_utils.to_glb(
                    outputs['gaussian'][0],
                    outputs['mesh'][0],
                    simplify=0.90,  # Reduce triangles more aggressively?
                    texture_size=1024,
                )

                # Export GLB
                glb_data.export(glb_path)
                log_info(f"GLB model saved to: {glb_path}")
                return glb_path

        except ImportError as e:
             log_error(f"Failed to import trellis/PIL: {e}. Cannot generate GLB.")
             return None
        except Exception as e:
            log_error(f"Failed to generate GLB model: {e}")
            flush_gpu_memory() # Clear memory on failure
            return None


    def _import_glb(self, glb_path: Path, asset_name: str, prompt: str):
        """Import the generated GLB into Blender and mark as asset."""
        log_info(f"Importing GLB: {glb_path}")
        try:
            # Import GLB
            bpy.ops.import_scene.gltf(filepath=str(glb_path))

            # Get imported objects (usually the last objects added)
            # This assumes the GLB contains a single main object or hierarchy root
            imported_objs = bpy.context.selected_objects
            if not imported_objs:
                 # Fallback: find objects with names matching the GLB file stem
                 glb_stem = glb_path.stem
                 imported_objs = [obj for obj in bpy.data.objects if glb_stem in obj.name and obj.select_get()]
                 if not imported_objs:
                     log_error("Could not identify imported GLB object.")
                     return

            # Assume the first selected object is the main one
            obj = imported_objs[0]
            unique_obj_name = get_unique_object_name(asset_name + "_3D") # Add suffix
            obj.name = unique_obj_name

            # Optional: Parent other imported parts to the main object if needed
            # for other_obj in imported_objs[1:]:
            #     other_obj.parent = obj

            # --- Mark as Asset ---
            obj.asset_mark()
            obj.asset_data.author = bl_info["author"]
            obj.asset_data.description = f"{prompt} (3D GLB Model)"
            # Add tags
            tag_list = ["Generated", "3D Asset", "GLB"]
            base_prompt_words = re.findall(r'\b\w+\b', prompt.lower())[:3]
            tag_list.extend([word.capitalize() for word in base_prompt_words])
            for tag_name in tag_list:
                obj.asset_data.tags.new(name=tag_name.strip())

            # Generate asset preview (requires rendering or thumbnail)
            # Using the 2D image as a placeholder preview for the GLB
            try:
                with bpy.context.temp_override(id=obj):
                    bpy.ops.ed.lib_id_load_custom_preview(filepath=str(glb_path.with_suffix(".png"))) # Try finding related PNG
            except Exception as e:
                 log_warning(f"Could not generate asset preview for GLB {obj.name}: {e}. Use 2D source as fallback.")
                 try:
                      # Fallback to the 2D source image path (might not exist if only GLB generated)
                      source_png = glb_path.with_name(glb_path.stem.replace("_3D", "") + "_part_1.png") # Guess source name
                      if not source_png.exists():
                           source_png = glb_path.with_suffix(".png") # Try replacing .glb with .png

                      if source_png.exists():
                           with bpy.context.temp_override(id=obj):
                               bpy.ops.ed.lib_id_load_custom_preview(filepath=str(source_png))
                      else:
                           log_warning("No suitable preview image found for GLB.")

                 except Exception as e2:
                      log_warning(f"Fallback preview generation failed: {e2}")


            log_info(f"Imported and marked GLB asset: {obj.name}")

        except Exception as e:
            log_error(f"Failed to import or mark GLB asset {glb_path}: {e}")

    # --- Execute Method ---
    def execute(self, context):
        # 0. Pre-flight Checks
#        if not self._ensure_dependencies():
#            return {'CANCELLED'}

        scene = context.scene
        input_props = scene.asset_gen_inputs
        output_dir = user_data_path() / "generated_assets"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get prompts
        prompts = []
        if input_props.input_type == "PROMPT":
            prompt = scene.asset_prompt.strip()
            if prompt:
                prompts.append(prompt)
            else:
                self.report({'ERROR'}, "Asset prompt is empty.")
                return {'CANCELLED'}
        else: # TEXT_BLOCK
            text_block_name = input_props.text_block_name
            if text_block_name == "NONE" or text_block_name not in bpy.data.texts:
                 self.report({'ERROR'}, "Invalid or no Text Block selected.")
                 return {'CANCELLED'}
            text_block = bpy.data.texts[text_block_name]
            prompts = [line.body.strip() for line in text_block.lines if line.body.strip()]
            if not prompts:
                 self.report({'ERROR'}, f"Text Block '{text_block_name}' is empty.")
                 return {'CANCELLED'}

        # --- Main Generation Loop ---
        total_prompts = len(prompts)
        log_info(f"Starting generation for {total_prompts} prompt(s).")
        wm = context.window_manager
        wm.progress_begin(0, total_prompts * 4) # Estimate steps: generate, remove bg, plane, glb

        generated_something = False
        asset_counter = 0

        try:
            for i, current_prompt in enumerate(prompts):
                # Update progress
                wm.progress_update(i * 4)
                self.report({'INFO'}, f"Processing prompt {i+1}/{total_prompts}: '{current_prompt[:50]}...'")

                # Determine base name for this asset
                # Use scene name if single prompt, derive from prompt line otherwise
                if total_prompts == 1 and scene.asset_name.strip():
                     base_asset_name = sanitize_filename(scene.asset_name.strip())
                else:
                     # Derive name from prompt (e.g., first few words)
                     derived_name = "_".join(re.findall(r'\b\w+\b', current_prompt)[:3])
                     base_asset_name = sanitize_filename(derived_name if derived_name else f"Asset_{i+1}")

                base_filename = base_asset_name # For file saving

                try:
                    # 1. Generate Raw Image
                    raw_image_path = self._generate_raw_image(current_prompt, output_dir, base_filename)
                    wm.progress_update(i * 4 + 1)

                    # 2. Remove Background
                    transparent_image_path = self._remove_background(raw_image_path, output_dir, base_filename)
                    wm.progress_update(i * 4 + 2)

                    # 3. Crop Transparent Image (optional but good)
                    cropped_path = self._crop_to_content(transparent_image_path)

                    # 4. Split into Parts (Islands)
                    # Use cropped path as input
                    split_image_paths = self._split_alpha_islands(cropped_path, output_dir, base_filename)

                    if not split_image_paths:
                         log_warning(f"No asset parts generated for prompt: '{current_prompt}'. Skipping.")
                         continue # Skip to next prompt if no content found


                    # 5. Process Each Part (Create Plane + Optionally GLB)
                    for part_index, part_path in enumerate(split_image_paths):
                        part_asset_name = f"{base_asset_name}_Part_{part_index+1}" if len(split_image_paths) > 1 else base_asset_name

                        # 5a. Create Blender Plane Asset
                        blender_obj = self._create_blender_plane(part_path, part_asset_name, current_prompt)
                        if blender_obj:
                            generated_something = True
                            asset_counter += 1
                        wm.progress_update(i * 4 + 3) # Increment progress slightly per part

                        # 5b. Generate and Import GLB Asset (Optional)
                        # Check if Trellis pipeline is available
                        if self._get_trellis_pipeline() is not None:
                             glb_file_path = self._generate_glb(part_path, output_dir, part_asset_name)
                             if glb_file_path:
                                 self._import_glb(glb_file_path, part_asset_name, current_prompt)
                                 generated_something = True # Count GLB import as success too
                                 # Note: GLB generation can be slow, progress update might need adjustment
                        wm.progress_update(i * 4 + 4) # Max progress for this prompt loop iteration


                    # Clean up intermediate files (optional)
                    if not DEBUG:
                         if raw_image_path.exists(): raw_image_path.unlink()
                         # Keep transparent/split files as they are used by Blender assets

                except Exception as e:
                    log_error(f"Failed processing prompt '{current_prompt}': {e}")
                    # Optionally report error to user for this specific prompt
                    self.report({'ERROR'}, f"Failed on prompt {i+1}: {e}. Continuing...")
                    # Ensure progress updates even on error
                    wm.progress_update((i + 1) * 4)
                    continue # Continue to the next prompt

            # End loop
            flush_gpu_memory() # Clear memory after all prompts are processed

        finally:
            # Ensure progress bar finishes
            wm.progress_end()
            # Clear cached models to free memory explicitly if addon might be disabled/re-enabled
            self._flux_pipe = None
            self._birefnet_model = None
            self._trellis_pipe = None
            log_info("Cleared cached models.")


        if generated_something:
            self.report({'INFO'}, f"Asset generation finished. Created {asset_counter} Blender plane asset(s). Check Asset Browser.")
            # Save the blend file to make assets persistent
            # bpy.ops.wm.save_mainfile() # Consider making this optional or a separate button
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "Asset generation finished, but no assets were successfully created.")
            return {'CANCELLED'}



# --- Panel ---
class ASSET_GENERATOR_PT_Panel(bpy.types.Panel):
    bl_label = "2D Asset Generator"
    bl_idname = "ASSET_GENERATOR_PT_Panel" # Convention: CATEGORY_PT_name
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "2D Asset Gen" # Keep category short

    @classmethod
    def poll(cls, context):
        # Optionally disable panel if dependencies aren't met?
        # prefs = get_addon_pref()
        # return prefs.dependencies_installed
        return True # Always show, operator handles dependency check

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        input_props = scene.asset_gen_inputs # Access through scene pointer

        # Dependency Check Info
        prefs = get_addon_pref()
        if not prefs.dependencies_installed:
             box = layout.box()
             box.label(text="Dependencies missing!", icon='ERROR')
             box.label(text="Install via Add-on Preferences.")
             row = box.row()
             row.operator(CheckDependenciesOperator.bl_idname, text="Check Again", icon='VIEWZOOM')
             row.operator("preferences.addon_show", text="Preferences").module = ADDON_ID
             layout.separator()
             # Disable generation button if deps not installed?
             # layout.enabled = False # Or just rely on operator check

        # Input Source Selection
        layout.prop(input_props, "input_type", expand=True)

        # Input Fields based on type
        box = layout.box()
        if input_props.input_type == "PROMPT":
            box.prop(scene, "asset_prompt", text="Prompt")
            # Name for the *single* asset generated from prompt
            box.prop(scene, "asset_name", text="Asset Name")
        else: # TEXT_BLOCK
            row = box.row(align=True)
            row.prop(input_props, "text_block_name", text="")
            # Display selected text block name read-only?
            # layout.label(text=f"Using: {input_props.text_block_name}")
            # Note: Asset names will be derived from lines in the text block

        # Generation Button
        layout.operator(ASSET_GENERATOR_OT_Generate.bl_idname, text="Generate Asset(s)", icon='PLAY')


# --- Helper Context Manager for Imports ---
# Ensures venv is in path temporarily during specific operations
# Useful if functions are called from contexts where sys.path might not be set yet
class VenvImportGuard:
    def __enter__(self):
        add_venv_to_sys_path()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        # No need to remove from sys.path typically, it just stays there
        pass


# --- Registration ---
classes = (
    AssetGeneratorPreferences,
    InstallDependenciesOperator,
    CheckDependenciesOperator,
    UninstallDependenciesOperator,
    AssetGenInputProps,
    ASSET_GENERATOR_OT_Generate,
    ASSET_GENERATOR_PT_Panel,
)

# ... (rest of register/unregister functions remain the same) ...

def register():
    log_info("Registering addon...")
    add_venv_to_sys_path() # Ensure path is added on registration

    for cls in classes:
        bpy.utils.register_class(cls)

    # Register Scene Properties
    bpy.types.Scene.asset_gen_inputs = PointerProperty(type=AssetGenInputProps)
    bpy.types.Scene.asset_prompt = StringProperty(
        name="Asset Prompt",
        description="Describe the asset to generate (used if Input Source is 'Prompt')",
        default="A cute cartoon cat character",
    )
    bpy.types.Scene.asset_name = StringProperty(
        name="Asset Name",
        description="Base name for the generated asset (if Input Source is 'Prompt')",
        default="MyAsset",
    )
    log_info("Addon registered successfully.")


def unregister():
    log_info("Unregistering addon...")
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as e:
            log_warning(f"Failed to unregister class {cls.__name__}: {e}")

    # Delete Scene Properties safely
    for prop_name in ["asset_gen_inputs", "asset_prompt", "asset_name"]:
         if hasattr(bpy.types.Scene, prop_name):
              try:
                   delattr(bpy.types.Scene, prop_name)
              except Exception as e:
                   log_warning(f"Failed to delete scene property {prop_name}: {e}")
         else:
              debug_print(f"Scene property {prop_name} not found during unregister.")
    log_info("Addon unregistered.")


# Required for Blender to run register/unregister
if __name__ == "__main__":
    register()
