def _normalize_cell(cell, room, r_dim=5):
    z, x, y = cell
    return z, x - room[0] * r_dim, y - room[1] * r_dim

def _id_room(cell, r_dim=5):
    _, x, y = cell

    room = (max(x // r_dim, 0), max(y // r_dim, 0))

    return room