"""User-supplied stable FWL partition for the official BREC pair ordering.

This table is supplied evidence, not recomputed or independently certified here.
Each run binds its use to the input dataset SHA256 in its manifest.
"""
POLICY = 'user_supplied_brec_fwl_270_380_v1'


def group_for_pair(pid):
    if not 0 <= pid < 400:
        raise ValueError('BREC pair ID must be in 0..399')
    if 340 <= pid < 360:
        return '3-FWL-indistinguishable'
    if 110 <= pid < 160 or 320 <= pid < 340 or 360 <= pid < 400:
        return '3-FWL-only'
    return '2-FWL'


def impossible(group, order):
    return group == '3-FWL-indistinguishable' or (
        order == 2 and group == '3-FWL-only')
