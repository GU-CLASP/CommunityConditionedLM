{ bootstrap ? import <nixpkgs> {} }:

let
    pkgs_source = fetchTarball "https://github.com/NixOS/nixpkgs/archive/dfa8e8b9bc4a18bab8f2c55897b6054d31e2c71b.tar.gz";
    overlays = [
      (self: super:  # define our local packages
         {
          python3 = super.python3.override {
           packageOverrides = python-self: python-super: {
             torchtext = python-self.callPackage /opt/nix/torchtext-0.4.0.nix { };
             snapy = python-self.callPackage /opt/nix/snapy-1.0.2.nix { };
             mmh3 = python-self.callPackage /opt/nix/mmh3-2.5.1.nix { };
             six = python-self.callPackage ./nix/six-1.14.0.nix { };
             google-resumable-media = python-self.callPackage ./nix/google-resumable-media-0.5.0.nix { };
             google-cloud-core = python-self.callPackage ./nix/google-cloud-core-1.3.0.nix { };
             google-api-core = python-self.callPackage ./nix/google-api-core-1.16.0.nix { };
             google-auth = python-self.callPackage ./nix/google-auth-1.11.3.nix { };
             googleapis-common-protos = python-self.callPackage ./nix/googleapis-common-protos-1.51.0.nix { };
             google-cloud-bigquery = python-self.callPackage ./nix/google-cloud-bigquery-1.24.0.nix { };
             google-cloud-bigquery-storage = python-self.callPackage ./nix/google-cloud-bigquery-storage-0.8.0.nix { };

           };};})
      ((import /opt/nix/nvidia-450.66.nix  ) pkgs_source )  # fix version of nvidia drivers
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
         #mmh3
         #snapy
         google-cloud-bigquery
         google-cloud-bigquery-storage
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
