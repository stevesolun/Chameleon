#!/usr/bin/env python3
"""
Clean GPT-5 answers in the comprehensive dataset:
1. Extract first letter (A, B, C, D) from any longer responses
2. Calculate is_correct field by comparing with correct_answer
3. Save the cleaned dataset
"""

import pandas as pd
import re
import numpy as np

def extract_letter_answer(answer):
    """
    Extract the first A, B, C, or D from a GPT-5 answer.
    
    Args:
        answer: The GPT-5 response
        
    Returns:
        str: Single letter A, B, C, or D, or original if not found
    """
    if pd.isna(answer) or answer == '':
        return answer
    
    # Convert to string and clean
    answer_str = str(answer).strip()
    
    # If already a single letter A, B, C, or D, return as is
    if answer_str in ['A', 'B', 'C', 'D']:
        return answer_str
    
    # Extract first occurrence of A, B, C, or D
    match = re.search(r'[ABCD]', answer_str)
    if match:
        return match.group()
    
    # If no A, B, C, D found, check for lowercase
    match = re.search(r'[abcd]', answer_str.lower())
    if match:
        return match.group().upper()
    
    # Return original if no valid answer found
    return answer_str

def calculate_is_correct(row):
    """
    Calculate if the GPT-5 answer is correct.
    
    Args:
        row: DataFrame row with gpt5_answer and correct_answer
        
    Returns:
        bool: True if correct, False if incorrect, NaN if no answer
    """
    if pd.isna(row['gpt5_answer']) or row['gpt5_answer'] == '':
        return np.nan
    
    if pd.isna(row['correct_answer']) or row['correct_answer'] == '':
        return np.nan
    
    return str(row['gpt5_answer']).strip() == str(row['correct_answer']).strip()

def main():
    """
    Main function to clean the dataset.
    """
    print('🧹 CLEANING GPT-5 ANSWERS')
    print('=' * 40)
    
    # Load the dataset
    input_file = 'distortions/chameleon_dataset.csv'
    print(f'📊 Loading dataset: {input_file}')
    
    df = pd.read_csv(input_file)
    print(f'   Total rows: {len(df):,}')
    print(f'   Total columns: {len(df.columns)}')
    
    # Check current state
    print('\n🔍 ANALYZING CURRENT STATE:')
    print(f'   Rows with GPT-5 answers: {df["gpt5_answer"].notna().sum():,}')
    print(f'   Empty is_correct fields: {df["is_correct"].isna().sum():,}')
    
    # Sample of current GPT-5 answers
    print(f'\\n📝 SAMPLE GPT-5 ANSWERS:')
    sample_answers = df['gpt5_answer'].dropna().head(10).tolist()
    for i, answer in enumerate(sample_answers, 1):
        print(f'   {i}. "{answer}"')
    
    # Check for problematic answers
    print(f'\\n🔍 CHECKING FOR PROBLEMATIC ANSWERS:')
    long_answers = df[df['gpt5_answer'].notna() & (df['gpt5_answer'].str.len() > 1)]
    print(f'   Answers longer than 1 character: {len(long_answers):,}')
    
    if len(long_answers) > 0:
        print('   Sample long answers:')
        for i, (_, row) in enumerate(long_answers.head(5).iterrows(), 1):
            print(f'     {i}. "{row["gpt5_answer"]}"')
    
    # Clean GPT-5 answers
    print(f'\\n🧹 CLEANING GPT-5 ANSWERS:')
    original_answers = df['gpt5_answer'].copy()
    df['gpt5_answer'] = df['gpt5_answer'].apply(extract_letter_answer)
    
    # Count changes
    changes = (original_answers != df['gpt5_answer']).sum()
    print(f'   Answers modified: {changes:,}')
    
    if changes > 0:
        print('   Sample changes:')
        changed_mask = original_answers != df['gpt5_answer']
        for i, (original, cleaned) in enumerate(zip(original_answers[changed_mask].head(5), 
                                                   df.loc[changed_mask, 'gpt5_answer'].head(5)), 1):
            print(f'     {i}. "{original}" → "{cleaned}"')
    
    # Calculate is_correct field
    print(f'\\n✅ CALCULATING IS_CORRECT:')
    df['is_correct'] = df.apply(calculate_is_correct, axis=1)
    
    # Summary statistics
    total_answered = df['gpt5_answer'].notna().sum()
    total_correct = df['is_correct'].sum()
    accuracy = (total_correct / total_answered * 100) if total_answered > 0 else 0
    
    print(f'   Total answered: {total_answered:,}')
    print(f'   Total correct: {total_correct:,}')
    print(f'   Accuracy: {accuracy:.1f}%')
    
    # Validation
    print(f'\\n🔍 VALIDATION:')
    valid_answers = df['gpt5_answer'].isin(['A', 'B', 'C', 'D']).sum()
    print(f'   Valid A/B/C/D answers: {valid_answers:,}')
    print(f'   Invalid answers: {total_answered - valid_answers:,}')
    
    # Show distribution
    print(f'\\n📊 ANSWER DISTRIBUTION:')
    answer_counts = df['gpt5_answer'].value_counts()
    for answer in ['A', 'B', 'C', 'D']:
        count = answer_counts.get(answer, 0)
        pct = (count / total_answered * 100) if total_answered > 0 else 0
        print(f'   {answer}: {count:,} ({pct:.1f}%)')
    
    # Save cleaned dataset
    output_file = 'distortions/comprehensive_distortion_dataset_CLEANED_20250922.csv'
    print(f'\\n💾 SAVING CLEANED DATASET:')
    print(f'   Output file: {output_file}')
    
    df.to_csv(output_file, index=False)
    print(f'   ✅ Saved {len(df):,} rows')
    
    print(f'\\n🎉 CLEANUP COMPLETE!')
    print(f'   Original file: {input_file}')
    print(f'   Cleaned file: {output_file}')
    print(f'   Accuracy: {accuracy:.1f}%')
    print(f'   Valid format rate: {(valid_answers/total_answered*100):.1f}%')

if __name__ == '__main__':
    main()
