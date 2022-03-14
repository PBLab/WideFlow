from colour import Color


def generate_gradient_color_list(n_colors, from_color="blue", to_color="green"):
    first_color = Color(from_color)
    last_color = Color(to_color)
    colors = list(first_color.range_to(last_color, n_colors))

    color_list = []
    for i in range(n_colors):
        color_list.append([
            colors[i].get_red(),
            colors[i].get_green(),
            colors[i].get_blue()
        ])

    return color_list
