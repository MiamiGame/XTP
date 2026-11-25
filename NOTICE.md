# Notice

XTP is a modified version of **VIctory** by Marvisak, imported from commit
`ffe310eb2098e52169f4c44db5ffdabf43a72622` (2026-11-24) and modified from
2026-11-24 onward. VIctory and XTP are distributed under the GNU General
Public License, version 3 or (at your option) any later version; see
[LICENSE](LICENSE).

The history of the original work is in the VIctory repository; the history of
the changes made for XTP is in this repository's git log.

Third-party code carried over from VIctory keeps its own copyright notices
and licences, stated at the top of the files concerned.

## Constructeur de structures d'accélération

- Paramètres racine déportés d'après Marvisak/VIctory (`27c3dc9d5`),
  fxceless's VIctory fork (`578d459`) et les tests de louketa dans
  Marvisak/VIctory#489 (`fe7210e`).
- Vues UAV racine d'après DuarttBR dans Marvisak/VIctory#1077
  (`873548f`). Accès par demi-mot d'après iurysf dans Marvisak/VIctory#1572
  (`0c16661`) et fxceless's VIctory fork (`4e17478`).
- Dimensions en threads et masque de fin de groupe d'après Yorivel
  (`2521646b8`) et mynzjerko (#646, `170c24269`) dans fxceless's VIctory fork.
- Admission du constructeur entier d'après 0xkn0wh3r3 dans
  Marvisak/VIctory#2118 (`60632d3f`, GPL-3.0).

## References that are not copied

The following projects are studied as references only. **No code is copied
from them**, because their licences are not compatible with XTP's, or because
they target another platform:

- **xcl** (<https://github.com/ivall/xcl>), GPL-2.0-only.
- **AnyPS5** (<https://github.com/simnJS/AnyPS5>), GPL-2.0. Only its
  contributor process inspires ours.
