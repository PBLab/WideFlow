def write_matching_point_file(path, match_p_src, match_p_dst):
    with open(path, 'w') as f:
        src_line, dst_line = 'match_p_src: [', 'match_p_dst: ['
        for src_p, dst_p in zip(match_p_src, match_p_dst):
            src_line += f'[{src_p[0]}, {src_p[1]}], '
            dst_line += f'[{dst_p[0]}, {dst_p[1]}], '
        src_line, dst_line = src_line[:-2] + ']\n', dst_line[:-2] + ']'

        f.writelines([src_line, dst_line])
