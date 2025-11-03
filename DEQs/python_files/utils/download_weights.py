import os
import subprocess
import sys

def ensure_gdown():
    """Check if gdown is installed, otherwise install it."""
    try:
        import gdown
    except ImportError:
        print("[INFO] Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    return gdown

# 🔧 Dictionary mapping all weights
_WEIGHTS_FILES = {
    'RED_Gauss_high':'1-yzChjutpH7vuOcIFrplb4vlINZrP3Rk',
    'RED_Gauss_medium':'1TZM8l7x1DAGzhyly83rTdKfc8_cuQBKF',
    'RED_Gauss_low': '1PYUz0tXnRYZMhgfmVVdwGIh_eiqF3KF6',
    
    'RED_Motion_7_high':'1ObjW2fIxgsIjLMA5E-N469wbsEjJjGrm',
    'RED_Motion_7_medium':'1WwJf10-l__IoefhH8EIyDUmhOrf5I02j',
    'RED_Motion_7_low':'1hdns6Mvdsx-AJiAYLbZJgi0grfmj_uny',
    
    'RED_Uniform_high':'1opZ1nl5ewDxZYdskkLJ83FhihTnqrdoC',
    'RED_Uniform_medium':'1lvhhJMztQyNeZOLPMkFuHuAPSPQnYVP4',
    'RED_Uniform_low':'1jo_LHYSbsImmJ8cBn_Mt2U6R17I1lm6E',
    
    'Scalar_Gauss_high':'1_qFuKuy6GeG8z7eKHa6z-lCJuBpo7fm1',
    'Scalar_Gauss_medium':'1NQzhae3FNV83zcwPMXkOP-KthsFREg8M',
    'Scalar_Gauss_low':'1TND0UQIMQCNDcFYQce7L6vUVuIunPkwL',
    
    'Scalar_Motion_7_high':'1fPJgHByiuh_kS_b60vpnUjA1QtZrDMpX',
    'Scalar_Motion_7_medium':'16Ce_Hpb8BqCRhzkv6hek9avcdx2yYen2',
    'Scalar_Motion_7_low':'15J1DVpkScQ3wBLueFz6iRCyXvHH01PSR',
    
    'Scalar_Uniform_high':'1Zw-SxcXeeVdGbUaxKbUZo-JJPiHB2FlI',
    'Scalar_Uniform_medium':'14a_7Cbx1R--LkoBVLvq5MZwloKfe28qi',
    'Scalar_Uniform_low':'1gKN0Vkjc4FluHrXM04lvNJhNJfTFi-k7'
}

def download_weight(regularization: str, kernel: str, intensity: str, weights_dir="weights") -> str:
    """
    Download the weight corresponding to the specified combination.

    Args:
        regularization: 'RED' or 'Scalar'
        kernel: 'Gauss', 'Motion_7', or 'Uniform'
        intensity: 'high', 'medium', or 'low'
        weights_dir: folder where weights will be saved

    Returns:
        Local path of the downloaded weight
    """
    gdown = ensure_gdown()
    os.makedirs(weights_dir, exist_ok=True)
    
    key = f"{regularization}_{kernel}_{intensity}"
    if key not in _WEIGHTS_FILES:
        raise ValueError(f"[ERROR] Invalid combination: {key}")
    
    file_id = _WEIGHTS_FILES[key]
    dest_path = os.path.join(weights_dir, key)
    
    if not os.path.exists(dest_path):
        print(f"[INFO] Downloading {key} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=False)
        print(f"[INFO] ✅ {key} downloaded to {dest_path}")
    else:
        print(f"[INFO] {key} already exists.")
    
    return dest_path
