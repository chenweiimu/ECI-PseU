ECL-PseU: effective classifier identifying-Pseudouridine sites

	The current version supports H. sapiens, S. cerevisiae and M. musculus three species.

	requirements£ºsklearn  0.20.3  pandas 0.23.4

usage:
	python ECL-PseU -i input.file -o output.file -s 0
	-s stand for sepice
	0 stand for H. sapiens
	1 stand for S. cerevisiae
	2 stand for M. musculus

example: 
	>P1
	GCUAAACAGGUACUGCUGGGC
	>P2
	UUAUUGAGUGUCUACUGUGUG
	>P3
	GAUAAACUGUUACGCAUAUAU
	>P4
	UUGUCGGUGUUAACAAAAUGG

results:
	P1 GCUAAACAGGUACUGCUGGGC       The U site at position [11] is pseudouridine!
	P2 UUAUUGAGUGUCUACUGUGUG       The U site at position [11] is pseudouridine!
	P3 GAUAAACUGUUACGCAUAUAU       The U site at position [11] is pseudouridine!
	P4 UUGUCGGUGUUAACAAAAUGG       The U site at position [] is pseudouridine!
	P5 UCGGGCCUAGUUCAAACCUUU       The U site at position [11] is pseudouridine!

