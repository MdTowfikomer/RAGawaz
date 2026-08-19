"""
Unit test for Ticket 1: MSMARCO-XI Hindi Dataset Schema Validation
"""
import pytest
import requests
import io
import pyarrow.parquet as pq


HINVAL_URL = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet"


@pytest.fixture(scope="module")
def parquet_schema():
    """Fetch parquet footer via HTTP range request (instant & lightweight)."""
    headers = {"Range": "bytes=-65536"}
    resp = requests.get(HINVAL_URL, headers=headers, allow_redirects=True)
    assert resp.status_code in [200, 206], f"Failed to fetch footer: {resp.status_code}"
    assert resp.content[-4:] == b"PAR1", "Invalid parquet file magic bytes"
    
    bio = io.BytesIO(resp.content)
    pf = pq.ParquetFile(bio)
    return pf


def test_schema_has_required_top_level_fields(parquet_schema):
    """Verify all critical top-level columns exist in schema."""
    arrow_names = parquet_schema.schema_arrow.names
    assert "query" in arrow_names, "Missing 'query' column"
    assert "query_id" in arrow_names, "Missing 'query_id' column"
    assert "Answer" in arrow_names, "Missing 'Answer' column"
    assert "passages" in arrow_names, "Missing 'passages' column"
    assert "target_lang" in arrow_names, "Missing 'target_lang' column"


def test_schema_has_translated_passages_nested_fields(parquet_schema):
    """Verify nested passages structure contains Translated_passages and is_selected."""
    schema_arrow = parquet_schema.schema_arrow
    passages_field = schema_arrow.field("passages")
    assert passages_field is not None, "Missing passages field in arrow schema"
    
    # Check subfields inside passages struct
    struct_type = passages_field.type
    subfield_names = [struct_type.field(i).name for i in range(struct_type.num_fields)]
    assert "Translated_passages" in subfield_names, "Missing 'Translated_passages' in passages"
    assert "is_selected" in subfield_names, "Missing 'is_selected' in passages"
    assert "English_passages" in subfield_names, "Missing 'English_passages' in passages"


def test_validation_split_has_sufficient_rows(parquet_schema):
    """Verify the validation split has ample rows to sample 50,000 passages."""
    num_rows = parquet_schema.metadata.num_rows
    # 97,941 rows with ~7-10 passages each = ~700k-1M passages
    assert num_rows >= 50000, f"Expected at least 50k rows, got {num_rows}"
