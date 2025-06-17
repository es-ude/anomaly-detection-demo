{ pkgs, lib, config, inputs, ... }:
let
  pkgs-unstable = import inputs.nixpkgs-unstable { system = pkgs.stdenv.system; };
in
{
  packages = [
    pkgs.git
  ];

  env.UV_VENV="${config.env.DEVENV_STATE}/venv";

  # env variables for anomaly detector
  # Override these in your local `devenv.local.nix` file
  # env.DEVICE="mps";
  # env.NUM_WORKERS=0;
  # env.IMAGE_WIDTH=128;
  # env.IMAGE_HEIGHT=128;
  # env.HISTORY_FILE_NAME="history.csv";
  # env.VERSION_FILE_NAME="commit_hash.txt";
  # env.MODEL_FILE_NAME="model.pt";
  # env.QUANT_MODEL_FILE_NAME="model_int8.pt";
  # env.MVTEC_DATASET_DIR="path/to/mvtec_ad_evaluation";
  # env.MVTEC_OBJECT="hazelnut";
  # env.MVTEC_OUTPUT_DIR="path/to/MVTec/mvtec_out";
  # env.MVTEC_CKPT_DIR="path/to/model_checkpoints/mvtec";
  # env.COOKIE_DATASET_DIR="path/to/Cookie/v2";
  # env.COOKIE_OUTPUT_DIR="path/to/Cookie/cookie_out";
  # env.COOKIE_CKPT_DIR="path/to/model_checkpoints/cookie";

  languages.python = {
    enable = true;
    version = "3.11";
    venv.enable = true;
    uv = {
      enable = true;
      package = pkgs-unstable.uv;
      # sync.enable = true;
      # sync.allGroups = true;
    };
  };

  scripts = {
    demo_picam = {
      exec = ''
        env ENABLE_PI_CAM=True devenv up
      '';
      description = "start demo with PiCam";
    };
    demo_webcam = {
      exec = ''
        if [[ -z "$1" ]]; then
          devenv up
        else
          env CAM_PORT="$1" devenv up
        fi
      '';
      description = "start demo with webcam (External on Notebook: 203)";
    };
  };

  processes = {
    demo = {
      exec = ''
          uv run src/demo_interface/demo.py
        '';
    };
  };

  enterShell = ''
    echo
    echo "Welcome Back to Cookie Detection"
    echo
    echo "Initializing virtual environment..."
    uv venv --system-site-packages --python ${pkgs.python3}/bin/python3 --clear --quiet && echo "Virtual environment initialized."
    uv sync --quiet && echo "Python packages synchronized."
    echo
  '';
}
