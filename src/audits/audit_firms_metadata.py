#!/usr/bin/env python3
# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
"""Read-only FIRMS CSV metadata audit. Standard library only.

Never changes the CSVs or model inputs. Preserves version codes as strings.
Writes small reports only to --output-dir. CSV distributions cannot establish
historical download-side filtering or the actual download date.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

DEFAULT_FILES = [
    r'data/RESTRICTED_INPUT_NOT_INCLUDED\04_validation_fire\2022\MODIS_Fire_2022_Global.csv',
    r'data/RESTRICTED_INPUT_NOT_INCLUDED\04_validation_fire\2023\MODIS_Fire_2023_Global.csv',
]
CATEGORICAL = ('satellite', 'instrument', 'version', 'daynight', 'type', 'confidence')
NUMERIC = ('latitude', 'longitude', 'brightness', 'confidence')
REQUIRED = ('latitude', 'longitude', 'acq_date', 'brightness')


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(path, progress_every, do_hash):
    report = {
        'file': str(path.resolve()), 'source_access': 'READ_ONLY',
        'file_size_bytes': path.stat().st_size,
        'download_date': 'NOT_ESTABLISHED_FROM_CSV',
        'download_side_filters': 'NOT_ESTABLISHED_FROM_CSV',
        'upstream_deduplication': 'NOT_ESTABLISHED_FROM_CSV',
        'duplicate_row_check': 'NOT_PERFORMED',
        'collection_mapping': 'NOT_INFERRED_FROM_NUMERIC_OR_STRING_VERSION_CODES',
    }
    counts = {field: collections.Counter() for field in CATEGORICAL}
    missing = collections.Counter()
    monthly = collections.Counter()
    dates = collections.Counter()
    joint = collections.Counter()
    numbers = {field: {'finite_rows': 0, 'invalid_or_missing_rows': 0,
                       'min': None, 'max': None} for field in NUMERIC}
    rows = 0
    malformed = 0
    invalid_dates = 0
    date_examples = []
    out_of_range_coordinates = 0
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        report['columns'] = headers
        report['missing_model_input_columns'] = [k for k in REQUIRED if k not in headers]
        for row in reader:
            rows += 1
            if None in row or any(value is None for key, value in row.items() if key is not None):
                malformed += 1
            for field in CATEGORICAL:
                value = (row.get(field) or '').strip()
                if not value:
                    missing[field] += 1
                else:
                    counts[field][value] += 1
            joint_key = tuple((row.get(k) or '').strip() for k in ('satellite', 'instrument', 'version'))
            joint[joint_key] += 1
            date_text = (row.get('acq_date') or '').strip()
            try:
                date = dt.date.fromisoformat(date_text)
                dates[date.isoformat()] += 1
                monthly[date.strftime('%Y-%m')] += 1
            except ValueError:
                invalid_dates += 1
                if len(date_examples) < 5 and date_text not in date_examples:
                    date_examples.append(date_text)
            parsed = {}
            for field in NUMERIC:
                value = (row.get(field) or '').strip()
                try:
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError('nonfinite')
                except ValueError:
                    numbers[field]['invalid_or_missing_rows'] += 1
                    continue
                parsed[field] = number
                stat = numbers[field]
                stat['finite_rows'] += 1
                stat['min'] = number if stat['min'] is None else min(stat['min'], number)
                stat['max'] = number if stat['max'] is None else max(stat['max'], number)
            if (('latitude' in parsed and not -90 <= parsed['latitude'] <= 90)
                    or ('longitude' in parsed and not -180 <= parsed['longitude'] <= 180)):
                out_of_range_coordinates += 1
            if progress_every > 0 and rows % progress_every == 0:
                print('FIRMS_AUDIT_PROGRESS file={} rows={:,}'.format(path.name, rows), flush=True)
    report.update({
        'rows': rows, 'malformed_rows': malformed,
        'categorical_counts': {k: dict(sorted(v.items())) for k, v in counts.items()},
        'categorical_missing_rows': {k: missing[k] for k in CATEGORICAL},
        'numeric_statistics': numbers,
        'observed_acquisition_date_min': min(dates) if dates else None,
        'observed_acquisition_date_max': max(dates) if dates else None,
        'observed_distinct_acquisition_dates': len(dates),
        'monthly_record_counts': dict(sorted(monthly.items())),
        'daily_record_counts': dict(sorted(dates.items())),
        'invalid_acquisition_dates': invalid_dates,
        'invalid_acquisition_date_examples': date_examples,
        'coordinate_out_of_range_rows': out_of_range_coordinates,
        'platform_instrument_version_counts': [
            {'satellite': k[0], 'instrument': k[1], 'version': k[2], 'rows': v}
            for k, v in sorted(joint.items())
        ],
    })
    if do_hash:
        print('FIRMS_AUDIT_HASHING={}'.format(path), flush=True)
        report['sha256'] = file_sha256(path)
    report['status'] = ('PASS_METADATA_EXTRACTED'
                        if rows and not report['missing_model_input_columns'] and not malformed
                        else 'CHECK_REQUIRED')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--files', nargs='+', default=DEFAULT_FILES)
    parser.add_argument('--output-dir', default=r'data/RESTRICTED_INPUT_NOT_INCLUDED\sanity_reports\firms_csv_metadata')
    parser.add_argument('--progress-every', type=int, default=200000)
    parser.add_argument('--no-hash', action='store_true')
    args = parser.parse_args()
    paths = [Path(name) for name in args.files]
    output_dir = Path(args.output_dir)
    report_paths = [output_dir / 'firms_csv_metadata_audit.json',
                    output_dir / 'firms_csv_metadata_summary.txt']
    if any(source.resolve() == destination.resolve() for source in paths for destination in report_paths):
        parser.error('An output path overlaps an input CSV; no operation performed.')
    if any(destination.exists() for destination in report_paths):
        parser.error('Reports already exist. Choose a new --output-dir; existing files will not be overwritten.')
    reports = []
    errors = []
    for index, path in enumerate(paths, 1):
        print('[{}/{}] Reading CSV without modification: {}'.format(index, len(paths), path), flush=True)
        try:
            reports.append(audit(path, args.progress_every, not args.no_hash))
        except (OSError, UnicodeError, csv.Error) as exc:
            errors.append({'file': str(path), 'error': '{}: {}'.format(type(exc).__name__, exc)})
    result = {
        'audit': 'FIRMS_CSV_METADATA_READ_ONLY',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'status': 'PASS_METADATA_EXTRACTED' if not errors and all(r['status'] == 'PASS_METADATA_EXTRACTED' for r in reports) else 'CHECK_REQUIRED',
        'files': reports, 'errors': errors,
        'interpretation': [
            'Version codes are literal CSV strings. Collection and SP/NRT identities still need a reliable product/version mapping or original download documentation.',
            'Observed date and spatial extents describe records in the supplied CSVs, not necessarily the original download request or completeness of coverage.',
            'Observed field distributions cannot establish whether records were screened or deduplicated upstream.',
            'No filesystem timestamp is interpreted as the historical download date.',
            'This audit does not change data, recreate fire inputs, or rerun models.',
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with report_paths[0].open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    lines = ['FIRMS CSV METADATA AUDIT', 'Status: ' + result['status'], '']
    for item in reports:
        lines.extend([
            'File: ' + item['file'],
            'Rows: {:,}'.format(item['rows']),
            'Acquisition dates: {} to {}'.format(item['observed_acquisition_date_min'], item['observed_acquisition_date_max']),
            'Distinct dates: {}'.format(item['observed_distinct_acquisition_dates']),
        ])
        for field in CATEGORICAL:
            lines.append(field + ': ' + json.dumps(item['categorical_counts'][field], ensure_ascii=False, sort_keys=True))
        for field in ('latitude', 'longitude', 'brightness', 'confidence'):
            lines.append(field + ' statistics: ' + json.dumps(item['numeric_statistics'][field], ensure_ascii=False, sort_keys=True))
        lines.append('Monthly counts: ' + json.dumps(item['monthly_record_counts'], sort_keys=True))
        lines.append('SHA256: ' + item.get('sha256', 'NOT_COMPUTED'))
        lines.append('')
    for error in errors:
        lines.append('ERROR: ' + json.dumps(error, ensure_ascii=False))
    lines.extend(['INTERPRETATION'] + result['interpretation'])
    with report_paths[1].open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    for path in report_paths:
        print('REPORT={}'.format(path.resolve()), flush=True)
    print('STATUS={}'.format(result['status']), flush=True)
    print('FIRMS_CSV_METADATA_AUDIT_COMPLETED', flush=True)
    return 0 if result['status'] == 'PASS_METADATA_EXTRACTED' else 2


if __name__ == '__main__':
    sys.exit(main())
