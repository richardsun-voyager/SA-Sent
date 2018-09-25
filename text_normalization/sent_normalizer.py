from modulenorm.modNormalize import normalize
from modulenorm.modTokenizing import tokenize
usenorm = normalize()
def text_norm(line):
    usenorm = normalize()
    try:
        text_norm = usenorm.enterNormalize(line) # normalisasi enter, 1 revw 1 baris
        text_norm = usenorm.lowerNormalize(text_norm) # normalisasi huruf besar ke kecil
        text_norm = usenorm.repeatcharNormalize(text_norm) # normalisasi titik yang berulang
        #text_norm = usenorm.linkNormalize(text_norm) # normalisasi link dalam text
        #text_norm = usenorm.spacecharNormalize(text_norm) # normalisasi spasi karakter
        text_norm = usenorm.ellipsisNormalize(text_norm) # normalisasi elepsis (…)

        #tok = tokenize() # panggil modul tokenisasi
        #text_norm = tok.WordTokenize(text_norm) # pisah tiap kata pada kalimat
        text_norm = text_norm.split()

        text_norm = usenorm.spellNormalize(text_norm) # cek spell dari kata perkata
        text_norm = usenorm.wordcNormalize(text_norm,2) 
    except:
        print('Error happened for line:' + line)
        text_norm = []
    return text_norm