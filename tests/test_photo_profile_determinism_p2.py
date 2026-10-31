"""Independent checks of deterministic ICC metadata using real Pillow/LCMS profiles.

Only the creation dates of genuine LCMS sRGB profiles are varied. No colour
conversion, ICC parser, image encoder or profile serializer is mocked.
"""
from __future__ import annotations

import hashlib
import io
import struct

import numpy as np
from PIL import Image, ImageCms
import pytest

from starstack import photo_render


@pytest.fixture
def genuine_srgb():
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def dated_profile(raw, date, *, flags=0, intent=0, identifier=b"\0" * 16):
    data = bytearray(raw)
    struct.pack_into(">6H", data, 24, *date)
    struct.pack_into(">I", data, 44, flags)
    struct.pack_into(">I", data, 64, intent)
    data[84:100] = identifier
    profile = ImageCms.ImageCmsProfile(io.BytesIO(bytes(data)))
    # We exercise the real LCMS serializer too; this exact roundtrip ensures
    # our independently varied header is not silently normalised by the fixture.
    assert profile.tobytes() == bytes(data)
    return profile


def supply(monkeypatch, profiles):
    sequence = iter(profiles)
    def factory(name):
        assert name == "sRGB"
        return next(sequence).profile
    monkeypatch.setattr(photo_render.ImageCms, "createProfile", factory)


def digest_id(data):
    # Build the hash input as disjoint spans, rather than copying production's
    # in-place header mutation. Flags, intent and ID are excluded from this hash.
    hashed = (data[:44] + bytes(4) + data[48:64] + bytes(4)
              + data[68:84] + bytes(16) + data[100:])
    assert len(hashed) == len(data)
    return hashlib.md5(hashed).digest()


def colour_grid():
    # Includes dark night-sky codes, exact black/white, saturated primary/secondary
    # colours, and intermediate colours; 17^3 distinct RGB input samples.
    levels = np.array([0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 255], np.uint8)
    samples = np.stack(np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1)
    return Image.fromarray(samples.reshape(17, 17*17, 3))


def test_different_real_profile_dates_and_ids_normalise_to_identical_bytes(genuine_srgb, monkeypatch):
    profiles = [dated_profile(genuine_srgb, (2001, 1, 1, 0, 0, 0), identifier=b"a" * 16),
                dated_profile(genuine_srgb, (2026, 9, 24, 23, 59, 59), identifier=b"b" * 16),
                dated_profile(genuine_srgb, (2035, 12, 31, 12, 34, 56), identifier=b"c" * 16)]
    assert len({p.tobytes() for p in profiles}) == 3
    supply(monkeypatch, profiles)
    outputs = [photo_render.srgb_icc() for _ in profiles]
    assert outputs[0] == outputs[1] == outputs[2]
    assert struct.unpack_from(">6H", outputs[0], 24) == (2026, 9, 24, 0, 0, 0)
    assert outputs[0][84:100] == digest_id(outputs[0])


def test_only_creation_date_and_profile_id_change_all_colour_tags_remain_exact(genuine_srgb, monkeypatch):
    original = dated_profile(genuine_srgb, (2001, 2, 3, 4, 5, 6), flags=3, intent=2,
                             identifier=bytes(range(16)))
    before = original.tobytes()
    supply(monkeypatch, [original])
    after = photo_render.srgb_icc()
    assert len(after) == len(before)
    permitted = set(range(24, 36)) | set(range(84, 100))
    assert all(a == b for index, (a, b) in enumerate(zip(before, after)) if index not in permitted)
    assert after[128:] == before[128:], "complete tag table and colour payloads must remain identical"
    assert struct.unpack_from(">I", after, 44)[0] == 3
    assert struct.unpack_from(">I", after, 64)[0] == 2
    assert after[84:100] == digest_id(after)
    reopened = ImageCms.ImageCmsProfile(io.BytesIO(after))
    assert reopened.tobytes() == after
    assert "srgb" in ImageCms.getProfileName(reopened).lower()
    assert ImageCms.getDefaultIntent(reopened) == 2
    assert reopened.profile.profile_id == after[84:100]


@pytest.mark.parametrize("intent", [0, 1, 2, 3])
def test_lcms_input_and_output_colour_transforms_are_pixel_identical(genuine_srgb, monkeypatch, intent):
    original = dated_profile(genuine_srgb, (2001, 2, 3, 4, 5, 6))
    lab = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB"))
    supply(monkeypatch, [original])
    normalised = ImageCms.ImageCmsProfile(io.BytesIO(photo_render.srgb_icc()))
    image = colour_grid()
    before_transform = ImageCms.buildTransformFromOpenProfiles(original, lab, "RGB", "LAB", renderingIntent=intent)
    after_transform = ImageCms.buildTransformFromOpenProfiles(normalised, lab, "RGB", "LAB", renderingIntent=intent)
    before = ImageCms.applyTransform(image, before_transform)
    after = ImageCms.applyTransform(image, after_transform)
    assert np.array_equal(np.asarray(before), np.asarray(after))
    before_reverse = ImageCms.buildTransformFromOpenProfiles(lab, original, "LAB", "RGB", renderingIntent=intent)
    after_reverse = ImageCms.buildTransformFromOpenProfiles(lab, normalised, "LAB", "RGB", renderingIntent=intent)
    assert np.array_equal(np.asarray(ImageCms.applyTransform(before, before_reverse)),
                          np.asarray(ImageCms.applyTransform(before, after_reverse)))


@pytest.mark.parametrize("format", ["JPEG", "TIFF"])
def test_image_encoder_identity_no_longer_depends_on_profile_timestamp(genuine_srgb, monkeypatch, format):
    first = dated_profile(genuine_srgb, (2026, 9, 24, 12, 0, 0))
    second = dated_profile(genuine_srgb, (2026, 9, 24, 12, 0, 1))
    image = colour_grid()
    def encoded(profile):
        stream = io.BytesIO()
        options = {"quality": 95} if format == "JPEG" else {}
        image.save(stream, format=format, icc_profile=profile, **options)
        return stream.getvalue()
    before = [encoded(p.tobytes()) for p in (first, second)]
    assert hashlib.sha256(before[0]).digest() != hashlib.sha256(before[1]).digest()
    supply(monkeypatch, [first, second])
    after = [encoded(photo_render.srgb_icc()) for _ in range(2)]
    assert after[0] == after[1]
    decoded = [np.asarray(Image.open(io.BytesIO(blob))) for blob in before + after]
    assert all(np.array_equal(decoded[0], pixels) for pixels in decoded[1:])
