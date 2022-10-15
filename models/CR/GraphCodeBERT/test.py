if __name__ == '__main__':
    files = []
    test_file_names = 'data/medium/test.buggy-fixed_s1.buggy,' \
                      'data/medium/test.buggy-fixed_s1.fixedZdata/medium/test.buggy-fixed_s2.buggy,' \
                      'data/medium/test.buggy-fixed_s2.fixedZdata/medium/test.buggy-fixed_s3.buggy,' \
                      'data/medium/test.buggy-fixed_s3.fixed'
    for x in str(test_file_names).split('Z'):
        files.append(x)
    print('hi')