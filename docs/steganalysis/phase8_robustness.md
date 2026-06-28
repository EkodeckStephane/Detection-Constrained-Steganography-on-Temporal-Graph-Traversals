# Phase 8/10 - Robustesse et ablations

Date de demarrage : 26 juin 2026.

## Resultat actuel

La phase 8 dispose d'une campagne executable de robustesse sur les enregistrements
apparies de la phase 7 :

- sous-echantillonnage a 80 %;
- troncature a 50 %;
- bruit sur temps inter-evenements;
- ablations de groupes de features : probabilite/rang, entropie, cold-start et
  temporalite locale.

Execution :

```powershell
python code\scripts\run_phase8_robustness.py
```

Sorties :

- `results/tables/phase8_robustness.json`
- `results/tables/phase8_robustness.csv`

## Verdict

Les attaques simples ne font pas remonter les AUC au-dessus du seuil
`AUC_ext <= 0,60`. Sous-echantillonnage, troncature et bruit de timestamp
conservent un pire cas autour de 0,584 sur T-Drive cellules. Les ablations
confirment que probabilite/rang et temporalite locale restent les principaux
signaux residuels, mais leur amplitude est maintenant faible au point
d'exploitation conservateur.

## Consequence

Les phases 7 et 8 ferment l'infrastructure d'evaluation et le blocage AUC
immediat pour le point d'exploitation conservateur. La limite restante est la
capacite : plusieurs datasets n'acceptent presque aucun bit sous ces seuils.
La suite doit optimiser le front debit/AUC sans utiliser le test final.
