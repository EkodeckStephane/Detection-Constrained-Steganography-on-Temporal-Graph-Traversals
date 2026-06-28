# Phase 5/10 - Encodage et decodage

Date de demarrage : 24 juin 2026.

## Statut scientifique

Cette phase dispose maintenant d'un **contrat executable de synchronisation**
avec AEAD standard et candidats variables. Les resultats produits par
`phase5_codec_baseline` peuvent verifier le decodage exact, le BER passif nul,
l'authentification AES-GCM et une approximation locale de la distorsion.

Avant revendication Q1, cette limite doit etre levee par :

- remplacement du codeur dyadique quantifie par un range coding ou
  distribution matching plus proche de `C_psi(. | h_i)`;
- tests sur candidats produits par le modele temporel complet, pas seulement
  sur des distributions controlees;
- caracterisation statistique de `D_KL(S || C_psi)` sur traces reelles;
- remplacement du code repetition-3 par BCH/LDPC ou justification explicite
  d'un code correcteur plus adapte.

## Resultat actuel

La phase 5 dispose d'un contrat executable pour la synchronisation Alice--Bob :

- protection du message par AES-GCM avec etiquette 128 bits;
- CRC32 pour detecter les divergences internes;
- code correcteur repetition-3 comme baseline passive minimale;
- couplage dyadique quantifie sur les candidats tries canoniquement;
- mesure locale en variation totale et KL en bits;
- test de synchronisation sur plusieurs ordres de candidats;
- decodage exact lorsque les candidats, largeurs de blocs et cle sont
  identiques.

Cette implementation sert a verrouiller les invariants de synchronisation et
remplace la precedente baseline HMAC-stream. Elle reste une baseline de codage,
pas encore le range coder final.

## Execution

```powershell
python code\scripts\run_phase5_codec.py
```

Sortie :

- `results/tables/phase5_codec_baseline.json`

## Limite restante avant article

Le couplage dyadique encode jusqu'a `floor(log2(|A|))` bits par transition et
mesure la distorsion locale en variation totale et KL. Il ne realise pas encore
un range coding optimal. Les resultats Phase 5 actuels sont donc admissibles
pour l'ingenierie interne et la synchronisation, mais insuffisants pour conclure
sur la capacite finale ou la furtivite.
