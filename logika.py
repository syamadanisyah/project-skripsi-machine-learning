a = 3
b = 5

#penjumlahan = a + b
#pengurangan = a - b
#perkalian = a * b
#print(penjumlahan)
#print(pengurangan)
#print(perkalian)

x = 1
y = 1
z = 2
n = 3

hasil = []

for i in range(x+1):
    for j in range(y+1):
        for k in range(z+1):
            if (i +j +k) != n :
                hasil.append([i, j, k])
    print(hasil)

    #nulis apa yak bingung dan bosen, tapi gapapa ini proses dopamin detox untuk
    #meningkatkan reseptor dopamin di otak supaya bisa lebih fokus dan produktif
    #dan bisa menyukai atau netral terhadap mengerjakan tugas atau pekerjaan yang sebelumnya di anggap membosankan
    #atau tidak menyenangkan, GOD LUCK

    #bosen njirr


    angka = [1, 2, 3, 4, 5]

total = 0
for x in angka:
    total = total + x

print(total)  # output: 15