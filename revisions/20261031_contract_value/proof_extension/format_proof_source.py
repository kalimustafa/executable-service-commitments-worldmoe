from pathlib import Path
p=Path(r'C:\Users\Kali\Desktop\电力市场\EnergyEconomics_Revision_20261031\revision\proof_extension\S29_uniform_and_materialised_witness.md')
s=p.read_text(encoding='utf-8-sig')
r={
 'an arrival in slot r excludes slot r+96':'an arrival in slot $r$ excludes slot $r+96$',
 'let b=(b_1,...,b_N) be the stored immediate-service reference, R_t its exact cumulative sum, and e(t) the first unsaturated slot at or after t. Let A(b) be':'let $b=(b_1,\\ldots,b_N)$ be the stored immediate-service reference, $R_t$ its exact cumulative sum, and $e(t)$ the first unsaturated slot at or after $t$. Let $\\mathcal A(b)$ be',
 'which generate b under':'which generate $b$ under',
 'larger than E.':'larger than $E$.',
 'for every a in A(b), the explicit construction below gives an execution x(a) in X_24(a) which completes all admitted work by N':'for every $a\\in\\mathcal A(b)$, the explicit construction below gives an execution $x(a)\\in\\mathcal X_{24}(a)$ which completes all admitted work by $N$',
 'for each a. Define the exact earliest-service queue Q_t=max(Q_{t-1}+a_t-1,0).':'for each $a$. Define the exact earliest-service queue $Q_t=\\max(Q_{t-1}+a_t-1,0)$.',
 'gives |Q_t-q_t|<=E.':'gives $|Q_t-q_t|\\le E$.',
 'At each stored clearing slot c, its exact pre-service queue is at most b_c+E<1. Thus the exact queue also clears at c.':'At each stored clearing slot $c$, its exact pre-service queue is at most $b_c+E<1$. Thus the exact queue also clears at $c$.',
 'exact completion by N.':'exact completion by $N$.',
 'that array from b.':'that array from $b$.',
 'For the selected a,':'For the selected $a$,',
 'Setting L_N=A_N selects':'Setting $L_N=A_N$ selects',
 'between ell_t and u_t.':'between $\\ell_t$ and $u_t$.',
 'increments lie in [0,1]':'increments lie in $[0,1]$',
 'Hence 0<=x_t(a)<=1.':'Hence $0\\le x_t(a)\\le1$.',
 'obeys L_t<=X_t(a)<=U_t':'obeys $L_t\\le X_t(a)\\le U_t$',
 'Consequently x(a) is in X_24(a),':'Consequently $x(a)\\in\\mathcal X_{24}(a)$,',
 'violation of bar X is at most E.':'violation of $\\bar X$ is at most $E$.',
 'For example, ell_t is a maximum':'For example, $\\ell_t$ is a maximum',
 'since bar X has increments':'since $\\bar X$ has increments',
 'at most bar X_t+E.':'at most $\\bar X_t+E$.',
 'gives u_t>=bar X_t-E.':'gives $u_t\\ge\\bar X_t-E$.',
 'Thus |X_t(a)-bar X_t|<=E for every prefix.':'Thus $|X_t(a)-\\bar X_t|\\le E$ for every prefix.',
 'slot cost is M_z-Lipschitz in work, with M_z=max_t[0.25 k_z PUE_zt(|p_zt|+2)].':'slot cost is $M_z$-Lipschitz in work, with $M_z=\\max_t[0.25 k_z\\mathrm{PUE}_{zt}(|p_{zt}|+2)]$.',
 'offset is tau_z(R_N-A_N), bounded in absolute value by 21E_z.':'offset is $\\tau_z(R_N-A_N)$, bounded in absolute value by $21E_z$.',
 'all four repair L1 norms':'all four repair $L^1$ norms'
}
for a,b in r.items():
 assert a in s,a
 s=s.replace(a,b)
p.write_text(s,encoding='utf-8')
