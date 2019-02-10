symDict = {
    "α":"\\alpha",
    "θ":"\\theta",
    "τ":"\\tau",
    "β":"\\beta",
    "π":"\\pi",
    "υ":"\\upsilon",
    "γ":"\\gamma",
    "$":"\\varpi",
    "φ":"\\phi",
    "δ":"\\delta",
    "κ":"\\kappa",
    "ρ":"\\rho",
    "":"\\epsilon",
    "λ":"\\lambda",
    "%":"\\varrho",
    "χ":"\\chi",
    "ε":"\\varepsilon",
    "μ":"\\mu",
    "σ":"\\sigma",
    "ψ":"\\psi",
    "ζ":"\\zeta",
    "ν":"\\nu",
    "ς":"\\varsigma",
    "ω":"\\omega",
    "η":"\\eta",
    "ξ":"\\xi",
    "=":"=",
    "<":"<",
    ">":">",
    "Γ":"\\Gamma",
    "Λ":"\\Lambda",
    "Σ":"\\Sigma",
    "Ψ":"\\Psi",
    "∆":"\\Delta",
    "Ξ":"\\Xi",
    "Υ":"\\Upsilon",
    "Ω":"\\Omega",
    "Θ":"\\Theta",
    "Π":"\\Pi",
    "Φ":"\\Phi",
    "∑":"\sum",
    "⋂":"\\bigcap",
    "⊙":"\bigodot",
    "∏":"\\prod",
    "⋃":"\\bigcup",
    "⊗":"\\bigotimes",
    "∐":"\\coprod",
    "⊔":"\\bigsqcup",
    "⊕":"\\bigoplus",
    "∫":"\\int",
    "∨":"\\bigvee",
    "⊎":"\\biguplus",
    "∮":"\\oint",
    "∧":"\\bigwedge"
}

for a in range(ord('a'), ord('z')):
    symDict[chr(a)] = chr(a)

for a in range(ord('A'), ord('Z')):
    symDict[chr(a)] = chr(a)

for a in range(ord('0'), ord('9')):
    symDict[chr(a)] = chr(a)

print(symDict["6"])