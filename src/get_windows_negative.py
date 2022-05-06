input_file_path = '/net/data.isilon/ag-cherrmann/projects/06_HIV_Microglia/data/tads/hg38/bed/K562_Rao_2014-raw_TADs.txt.bed'
output_file_path = '/net/data.isilon/ag-cherrmann/echernova/tad_boundaries_10_40kb/K562_TAD_boundaries_10_40kb.txt.bed'

bin_number = 10
bin_size = 40000 # only even

input_f = open(input_file_path, 'r')
output_file = open(output_file_path, 'w')

chromosome_ends = {'chr1':248956422, 'chr2':242193529, 'chr3':198295559, 'chr4':190214555,
    'chr5':181538259, 'chr6':170805979, 'chr7':159345973, 'chrX':156040895, 'chr8':145138636,
    'chr9':138394717, 'chr11':135086622, 'chr10':133797422, 'chr12':133275309, 'chr13':114364328,
    'chr14':107043718, 'chr15':101991189, 'chr16':90338345, 'chr17':83257441, 'chr18':80373285,
    'chr20':64444167, 'chr19':58617616, 'chrY':57227415, 'chr22':50818468, 'chr21':46709983}

counter = 0
for line in input_f:
    counter += 1
    chrom = line.strip().split('\t')[0]
    central_u_border = line.strip().split('\t')[1]
    central_d_border = line.strip().split('\t')[2]

    border = (central_u_border+central_d_border)//2
    first_bin_u_border = border - bin_size//2 - bin_number*bin_size # upstream border of the first bin 

    for i in range(bin_number*2+1):
        bin_b = first_bin_u_border+bin_size*i
        if bin_b >= 0 and chromosome_ends[chrom]>=bin_b+bin_size:
            output_file.write("{chr}\t{u_border}\t{d_border}\t{window}\n".format(chr=chrom, 
                                                                u_border=bin_b, 
                                                                d_border=bin_b+bin_size,
                                                                window=i-bin_number))
print("Processed TAD boundaries: {}".format(counter)) 
input_f.close()
output_file.close()