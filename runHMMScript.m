%runHMM

for noS = [2,4,8]
    for noM = [2,4,8,16]
        newMainHMM(noS,noM)
    end
end