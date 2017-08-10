# t = os.walk(data_dir)
# for root,dir,files in os.walk(data_dir):
#
#     for file in files:
#             fname = root+'/'+file
#             if fname.split('.')[-1] == 'append':
#                 continue
#             print fname
#             fa = open(fname,'r')
#             fbname = fname + '.append'
#             fb = open(fbname,'w+')
#             while fa is not None:
#                 line = fa.readline()
#                 if line == '':
#                     break
#                 if line == '\n':
#                     new_line = '\n'
#                 else:
#                     segs = line.split('\t')
#                     assert len(segs) == 2
#                     if segs[1] == 'B\n':
#                         new_line = segs[0]+ '\t'+ 'NA\n'
#                     elif segs[1] == 'M\n':
#                         new_line = segs[0] +'\t'+'AP\n'
#                     elif segs[1] == 'S\n':
#                         new_line = segs[0] + '\t'+'NA\n'
#                     elif segs[1] == 'E\n':
#                         new_line = segs[0] + '\t' + 'AP\n'
#                 fb.write(new_line)
#             fa.close()
#             fb.close()