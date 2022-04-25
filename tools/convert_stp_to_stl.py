

import sys

import sys
import subprocess
from shutil import which
import platform
import os.path


def is_executable(name):
    """Check whether `name` is on PATH and marked as executable.
    for python3 only, but cross-platform"""

    # from whichcraft import which
    return which(name) is not None


def detect_lib_path(out, libname):
    """parse ldd output and extract the lib, POSIX only
    OSX Dynamic library naming:  lib<libname>.<soversion>.dylib
    """
    # print(type(out))
    output = out.decode("utf8").split("\n")
    for l in output:
        if l.find(libname) >= 0:
            # print(l)
            i_start = l.find("=> ") + 3
            i_end = l.find(" (") + 1
            lib_path = l[i_start:i_end]
            return lib_path
    print("dynamic lib file is not found, check the name (without suffix)")


def get_lib_dir(program, libname):
    program_full_path = which(program)
    # print(program_full_path)
    if is_executable("ldd"):
        process = subprocess.Popen(
            ["ldd", program_full_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = process.communicate()
        solib_path = detect_lib_path(out, libname)
        lib_path = os.path.dirname(solib_path)
        if os.path.exists(lib_path):
            return lib_path
        else:
            print("library file " + libname + " is found, but lib dir does not exist")
    else:
        print("ldd is not available, it is not posix OS")
        # macos: has no ldd, https://stackoverflow.com/questions/45464584/macosx-which-dynamic-libraries-linked-by-binary


def get_freecad_app_name():
    ""
    if is_executable("freecad"):  # debian
        return "freecad"
    elif is_executable("FreeCAD"):  # fedora
        return "FreeCAD"
    elif is_executable("freecad-daily"):
        return "freecad-daily"
    else:
        print("FreeCAD is not installed or set on PATH")
        return None


def get_freecad_lib_path():
    ""
    os_name = platform.system()
    fc_name = get_freecad_app_name()
    if os_name == "Linux":
        if fc_name:
            return get_lib_dir(fc_name, "libFreeCADApp")
        else:
            return None
    elif os_name == "Windows":
        return get_lib_path_on_windows(fc_name)
    elif os_name == "Darwin":
        raise NotImplementedError("MACOS is not supported yet")
    else:
        # assuming POSIX platform with ldd command
        return get_lib_dir(fc_name, "libFreeCADApp")


def get_lib_path_on_windows(fc_name):
    # windows installer has the options add freecad to PYTHONPATH
    # this can also been done manually afterward, settting env variable PYTHONPATH
    # the code below assuming freecad is on command line search path

    if fc_name:
        fc_full_path = which(fc_name)
        lib_path = os.path.dirname(os.path.dirname(fc_full_path)) + os.path.sep + "lib"
        if os.path.exists(lib_path + os.path.sep + "Part.pyd"):
            return lib_path
    else:  # check windows registry key, if installed by installer
        get_installaton_path_on_windows("FreeCAD")


def get_installaton_path_on_windows(fc_name):
    # tested for FreeCAD 0.18
    import itertools
    from winreg import (
        ConnectRegistry,
        HKEY_LOCAL_MACHINE,
        OpenKeyEx,
        QueryValueEx,
        CloseKey,
        KEY_READ,
        EnumKey,
        WindowsError,
    )

    try:
        root = ConnectRegistry(None, HKEY_LOCAL_MACHINE)
        reg_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
        akey = OpenKeyEx(root, reg_path, 0, KEY_READ)
        for i in itertools.count():
            try:
                subname = EnumKey(akey, i)
            except WindowsError:
                break
            if subname.lower().find(fc_name.lower()) > 0:
                subkey = OpenKeyEx(akey, subname, 0, KEY_READ)
                pathname, regtype = QueryValueEx(subkey, "InstallLocation")
                CloseKey(subkey)
                return os.path.expandvars(pathname)
        # close key and root
        CloseKey(akey)
        CloseKey(root)
        return None
    except OSError:
        return None


def append_freecad_mod_path():
    try:
        import FreeCAD  # PYTHONPATH may has been set, do nothing
    except ImportError:
        # in case PYTHONPATH is not set
        cmod_path = get_freecad_lib_path()  # c module path
        if cmod_path:
            pymod_path = os.path.join(cmod_path, os.pardir) + os.path.sep + "Mod"
            sys.path.append(cmod_path)
            sys.path.append(pymod_path)

sys.path.append(get_freecad_lib_path())
import FreeCAD
# import Import
import FreeCADGui
FreeCADGui.showMainWindow()
import Part
import Mesh
import glob
import os 
import ImportGui
from tqdm import tqdm
import sys



dataset_path = sys.argv[1]

for product_path in tqdm(sorted(glob.glob(dataset_path + "/*"))):

    output_path = product_path + "/stl"
    if os.path.exists(output_path):
        print("Already exists at {}! SKipping this ...".format(product_path))
        continue
    cad_name = product_path.split("/")[-1]
    cad_file = product_path + "/tree.stp"
    print("Processing", cad_file)

    doc = FreeCAD.newDocument("doc") 
    FreeCAD.setActiveDocument(doc.Name)
    ImportGui.insert(cad_file, doc.Name) 
    
    for obj in FreeCAD.ActiveDocument.Objects:
        if hasattr(obj, "Shape"):
            label = obj.Label

            print("==> Saving", label)
            if not os.path.exists(output_path):
                cad_name = cad_name.split("_")[0]
                os.mkdir(output_path)
                print("Output path", output_path)
            Mesh.export([obj], output_path + "/" + label + ".stl")
        FreeCAD.closeDocument("doc")