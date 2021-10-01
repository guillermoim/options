def _normalize_cell(cell, room, r_dim=5):
    z, y, x = cell
    return z, y - room[0] * r_dim, x - room[1] * r_dim

def _de_normalize_cell(n_cell, room, r_dim=5):
    z, y, x = n_cell
    return z, y + room[0] * r_dim, x + room[1] * r_dim


def _id_room(cell, r_dim=5):
    _, x, y = cell

    room = (max(x // r_dim, 0), max(y // r_dim, 0))

    return room