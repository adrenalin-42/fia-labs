from production import IF, AND, THEN, OR, DELETE, NOT, FAIL

MY_RULES = (
    IF( AND( '(?x) wears expensive suit',          # R1
             '(?x) is tall' ),
        THEN( '(?x) has Earth Origin' )),

    IF( AND( '(?x) wears flashy gear',             # R2
             '(?x) has awkward gait' ),
        THEN( '(?x) has Earth Origin' )),

    IF( AND( '(?x) wears earth casual',            # R3
             '(?x) is tall' ),
        THEN( '(?x) has Earth Origin' )),

    IF( AND( '(?x) wears mars clothing',           # R4
             '(?x) has awkward gait' ),
        THEN( '(?x) has Space Origin' )),

    IF( AND( '(?x) wears zero-g clothing',         # R5
             '(?x) speaks belt slang' ),
        THEN( '(?x) has Space Origin' )),

    IF( AND( '(?x) speaks excited speech',         # R6
             '(?x) takes photos' ),
        THEN( '(?x) shows Tourist Behavior' )),

    IF( AND( '(?x) has camera',                    # R7
             '(?x) complains' ),
        THEN( '(?x) shows Tourist Behavior' )),

    IF( AND( '(?x) speaks corporate speech',       # R8
             '(?x) has briefcase' ),
        THEN( '(?x) is Professional' )),

    IF( AND( '(?x) checks time',                   # R9
             '(?x) wears expensive suit' ),
        THEN( '(?x) is Professional' )),

    IF( AND( '(?x) speaks academic speech',        # R10
             '(?x) asks questions' ),
        THEN( '(?x) is Academic' )),

    IF( AND( '(?x) has datapad',                   # R11
             '(?x) wears earth casual' ),
        THEN( '(?x) is Academic' )),

    IF( AND( '(?x) has Earth Origin',              # R12
             '(?x) is Professional' ),
        THEN( '(?x) is Earth Business Executive' )),

    IF( AND( '(?x) has Earth Origin',              # R13
             '(?x) shows Tourist Behavior' ),
        THEN( '(?x) is Earth Adventure Tourist' )),

    IF( AND( '(?x) has Earth Origin',              # R14
             '(?x) is Academic' ),
        THEN( '(?x) is Earth Academic' )),

    IF( AND( '(?x) has Space Origin',              # R15
             '(?x) asks questions' ),
        THEN( '(?x) is Mars Colonist' )),

    IF( AND( '(?x) has Space Origin',              # R16
             '(?x) has mining tools' ),
        THEN( '(?x) is Belt Miner' )),

    IF( OR( '(?x) wears lunar clothing',           # R17
            '(?x) speaks lunar dialect',
            '(?x) has smooth gait',
            '(?x) is short' ),
        THEN( '(?x) is Loonie' )),
)
