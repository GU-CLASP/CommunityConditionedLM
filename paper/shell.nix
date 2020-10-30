# Non-reproducible "current" version:
# with (import <nixpkgs> {});
with import (fetchTarball https://github.com/NixOS/nixpkgs-channels/archive/nixos-20.03.tar.gz) {};


let orgEmacs = emacsWithPackages (with emacsPackagesNg; [org]);
    ghc = haskellPackages.ghcWithPackages (hp: with hp; [mtl split]);
in stdenv.mkDerivation {
  name = "docsEnv";
  shellHook = ''
        export LANG=en_US.UTF-8
       '';
        # eval $(egrep ^export ${ghc}/bin/ghc)
  buildInputs = [ # orgEmacs
                  # haskellPackages.lhs2tex
                  # ghc
                  biber
                  zip
                  (texlive.combine {
      inherit (texlive)
      gb4e # terrible linguistic packagev
      # lstcoq # listing configuration for Coq snippets
      # collection-fontutils ?
      # dvipng # org-mode preview wants this; but broken
      arabtex 
      biber
      biblatex
      capt-of
      cleveref # clever "ref"
      cm-super
      collection-fontsrecommended
      comment
      dejavu # font
      doublestroke # usepackage dsfont
      dvipng # org-mode preview needs this
      dvisvgm # or even beter, this!
      ebgaramond # font
      enumitem
      environ # ???
      fancyhdr # fancy headers
      filehook
      fontaxes
      hyperref
      hyphenat # don't hyphenate
      inconsolata # font
      lastpage
      latexmk # make tool
      lazylist # for lhs2tex
      libertine # font
      libertinus # font
      lm # font, latin modern
      lm-math
      logreq
      make4ht # html conversion 
      marvosym
      mathdesign # jlm requirement
      mathpartir # math paragraph and inferrules (Didier Rémy)
      multirow
      newtx # newtxmath
      newunicodechar
      pgfplots
      polytable # for lhs2tex
      scheme-small
      siunitx
      soul # underline?
      stmaryrd # St Mary symbols
      # subcaption # subfigure is deprecated
      tex-gyre # font
      tex-gyre-math # font
      tex4ht  # html conversion
      threeparttable
      titlesec # change the environment of sections, etc. in convenient way
      titling # tweaking title
      tikz-dependency # Universal dependencies (NLP)
      todonotes
      ucharcat
      unicode-math
      varwidth
      wasy
      wasysym
      wrapfig
      xargs
      xstring;
    })
                ];
}
