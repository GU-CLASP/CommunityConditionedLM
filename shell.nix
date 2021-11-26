{ bootstrap ? import <nixpkgs> {} }:

let
    pkgs_source = fetchTarball "https://github.com/NixOS/nixpkgs/archive/dfa8e8b9bc4a18bab8f2c55897b6054d31e2c71b.tar.gz";
    overlays = [
      (self: super:  # define our local packages
         {
          python3 = super.python3.override {
           packageOverrides = python-self: python-super: {
             torchtext = python-self.callPackage /opt/nix/torchtext-0.4.0.nix { };
           };};})
      ((import /opt/nix/nvidia-current.nix  ) pkgs_source )  # fix version of nvidia drivers
      (self: super: {
          cudatoolkit = super.cudatoolkit_10; # fix version of cuda
          cudnn = super.cudnn_cudatoolkit_10;})
    ];
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
    pkgs = import pkgs_source {inherit overlays; inherit config;};
    py = pkgs.python3;
    pyEnv = py.buildEnv.override {
      extraLibs = with py.pkgs;
        [
         pytorch
         torchtext
         notebook
         matplotlib
         pandas
         scikitlearn
         plotly
         statsmodels
         numpy
        ];
      ignoreCollisions = true;};
in
  pkgs.stdenv.mkDerivation {
    name = "sh-env";
    buildInputs = [pyEnv pkgs.htop];
    shellHook = ''
      export LANG=en_US.UTF-8
      export PYTHONIOENCODING=UTF-8
      export LD_PRELOAD=/lib64/libcuda.so.1
     '';
  }
