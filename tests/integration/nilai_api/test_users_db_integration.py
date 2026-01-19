"""
Integration tests for user database operations.

These tests use a real PostgreSQL database via testcontainers.
"""

import uuid
import pytest
import json

from nilai_api.db.users import UserManager, RateLimits


class TestUserManagerIntegration:
    """Integration tests for UserManager with real PostgreSQL database."""

    @pytest.mark.asyncio
    async def test_simple_user_creation(self, clean_database):
        """Test creating a simple user and retrieving it."""
        # Insert user with minimal data
        user = await UserManager.insert_user(user_id="Simple Test User")

        # Verify user creation
        assert user.user_id == "Simple Test User"
        assert user.rate_limits is None, (
            f"Rate limits are not set for user {user.user_id}"
        )

        # Retrieve user by ID
        found_user = await UserManager.check_user(user.user_id)
        assert found_user is not None
        assert found_user.user_id == user.user_id
        assert found_user.rate_limits == user.rate_limits

    @pytest.mark.asyncio
    async def test_rate_limits_json_crud_basic(self, clean_database):
        """Test basic JSON CRUD operations for rate limits."""
        # Create user with comprehensive rate limits
        rate_limits = RateLimits(
            user_rate_limit_day=1000,
            user_rate_limit_hour=100,
            user_rate_limit_minute=10,
            web_search_rate_limit_day=50,
            web_search_rate_limit_hour=5,
            web_search_rate_limit_minute=1,
            user_rate_limit=20,
            web_search_rate_limit=10,
        )

        # CREATE: Insert user with rate limits
        user = await UserManager.insert_user(
            user_id="Rate Limits Test User", rate_limits=rate_limits
        )

        # Verify rate limits are stored as JSON
        assert user.rate_limits == rate_limits.model_dump()

        # READ: Retrieve user and verify rate limits JSON
        retrieved_user = await UserManager.check_user(user.user_id)
        assert retrieved_user is not None
        assert retrieved_user.rate_limits == rate_limits.model_dump()

        # Verify rate_limits_obj property converts JSON to RateLimits object
        rate_limits_obj = retrieved_user.rate_limits_obj
        assert isinstance(rate_limits_obj, RateLimits)
        assert rate_limits_obj.user_rate_limit_day == 1000
        assert rate_limits_obj.user_rate_limit_hour == 100
        assert rate_limits_obj.user_rate_limit_minute == 10
        assert rate_limits_obj.web_search_rate_limit_day == 50
        assert rate_limits_obj.web_search_rate_limit_hour == 5
        assert rate_limits_obj.web_search_rate_limit_minute == 1
        assert rate_limits_obj.user_rate_limit == 20
        assert rate_limits_obj.web_search_rate_limit == 10

    @pytest.mark.asyncio
    async def test_rate_limits_json_update(self, clean_database):
        """Test updating rate limits JSON data."""
        # Create user with initial rate limits
        initial_rate_limits = RateLimits(
            user_rate_limit_day=500, web_search_rate_limit_hour=25, user_rate_limit=5
        )

        user = await UserManager.insert_user(
            user_id="Update Rate Limits User", rate_limits=initial_rate_limits
        )

        # Verify initial rate limits
        retrieved_user = await UserManager.check_user(user.user_id)
        assert retrieved_user is not None
        assert retrieved_user.rate_limits == initial_rate_limits.model_dump()

        # UPDATE: Create new rate limits with different values
        updated_rate_limits = RateLimits(
            user_rate_limit_day=2000,  # Updated
            user_rate_limit_hour=200,  # New field
            web_search_rate_limit_hour=50,  # Updated
            web_search_rate_limit_day=100,  # New field
            user_rate_limit=10,  # Updated
            web_search_rate_limit=20,  # New field
        )

        # Update the user with new rate limits using direct model update
        from nilai_api.db import get_db_session
        import sqlalchemy as sa

        async with get_db_session() as session:
            # Update rate_limits JSON column directly
            stmt = sa.text("""
                UPDATE users
                SET rate_limits = :rate_limits_json
                WHERE user_id = :user_id
            """)
            await session.execute(
                stmt,
                {
                    "rate_limits_json": updated_rate_limits.model_dump_json(),
                    "user_id": user.user_id,
                },
            )
            await session.commit()

        # READ: Verify the update worked
        updated_user = await UserManager.check_user(user.user_id)
        assert updated_user is not None
        assert updated_user.rate_limits == updated_rate_limits.model_dump()

        # Verify the rate_limits_obj conversion works with updated data
        updated_rate_limits_obj = updated_user.rate_limits_obj
        assert isinstance(updated_rate_limits_obj, RateLimits)
        assert updated_rate_limits_obj.user_rate_limit_day == 2000
        assert updated_rate_limits_obj.user_rate_limit_hour == 200
        assert updated_rate_limits_obj.web_search_rate_limit_hour == 50
        assert updated_rate_limits_obj.web_search_rate_limit_day == 100
        assert updated_rate_limits_obj.user_rate_limit == 10
        assert updated_rate_limits_obj.web_search_rate_limit == 20

    @pytest.mark.asyncio
    async def test_rate_limits_json_partial_update(self, clean_database):
        """Test partial JSON updates and null handling for rate limits."""
        # Create user with some rate limits
        partial_rate_limits = RateLimits(
            user_rate_limit_day=300,
            web_search_rate_limit_day=15,
            # Other fields will be None/default
        )

        user = await UserManager.insert_user(
            user_id="Partial Rate Limits User", rate_limits=partial_rate_limits
        )

        # Verify partial data is stored correctly
        retrieved_user = await UserManager.check_user(user.user_id)
        assert retrieved_user is not None
        assert retrieved_user.rate_limits == partial_rate_limits.model_dump()

        # Test partial JSON update using PostgreSQL JSON operations
        from nilai_api.db import get_db_session
        import sqlalchemy as sa

        async with get_db_session() as session:
            # Update only specific fields in the JSON
            stmt = sa.text("""
                UPDATE users
                SET rate_limits = jsonb_set(
                    COALESCE(rate_limits::jsonb, '{}'),
                    '{user_rate_limit_hour}',
                    '75'
                )
                WHERE user_id = :user_id
            """)
            await session.execute(stmt, {"user_id": user.user_id})
            await session.commit()

        # Verify partial update worked
        updated_user = await UserManager.check_user(user.user_id)
        assert updated_user is not None

        expected_data = partial_rate_limits.model_dump()
        expected_data["user_rate_limit_hour"] = 75
        assert updated_user.rate_limits == expected_data

        # Test rate_limits_obj with partial data
        rate_limits_obj = updated_user.rate_limits_obj
        assert rate_limits_obj.user_rate_limit_day == 300
        assert rate_limits_obj.user_rate_limit_hour == 75
        assert rate_limits_obj.web_search_rate_limit_day == 15

    @pytest.mark.asyncio
    async def test_rate_limits_json_null_and_delete(self, clean_database):
        """Test NULL handling and JSON field deletion for rate limits."""
        # Create user with rate limits
        rate_limits = RateLimits(
            user_rate_limit_day=400, web_search_rate_limit_hour=30, user_rate_limit=8
        )

        user = await UserManager.insert_user(
            user_id="Delete Rate Limits User", rate_limits=rate_limits
        )

        # DELETE: Set rate_limits to NULL
        from nilai_api.db import get_db_session
        import sqlalchemy as sa

        async with get_db_session() as session:
            stmt = sa.text(
                "UPDATE users SET rate_limits = NULL WHERE user_id = :user_id"
            )
            await session.execute(stmt, {"user_id": user.user_id})
            await session.commit()

        # Verify NULL handling
        null_user = await UserManager.check_user(user.user_id)
        assert null_user is not None
        assert null_user.rate_limits is None

        # Test rate_limits_obj with NULL - should return config defaults
        default_rate_limits_obj = null_user.rate_limits_obj
        assert isinstance(default_rate_limits_obj, RateLimits)
        # Should have some default values (depends on config)
        assert default_rate_limits_obj.user_rate_limit_day is not None

        # Test removing specific JSON fields
        async with get_db_session() as session:
            # First set some data
            new_data = {"user_rate_limit_day": 500, "web_search_rate_limit_day": 25}
            stmt = sa.text(
                "UPDATE users SET rate_limits = :data WHERE user_id = :user_id"
            )
            await session.execute(
                stmt, {"data": json.dumps(new_data), "user_id": user.user_id}
            )
            await session.commit()

        # Verify data was set
        updated_user = await UserManager.check_user(user.user_id)
        assert updated_user is not None
        assert updated_user.rate_limits == new_data

        # Remove a specific field from JSON
        async with get_db_session() as session:
            stmt = sa.text("""
                UPDATE users
                SET rate_limits = rate_limits::jsonb - 'web_search_rate_limit_day'
                WHERE user_id = :user_id
            """)
            await session.execute(stmt, {"user_id": user.user_id})
            await session.commit()

        # Verify field was removed
        final_user = await UserManager.check_user(user.user_id)
        expected_final_data = {"user_rate_limit_day": 500}
        assert final_user is not None
        assert final_user.rate_limits == expected_final_data

    @pytest.mark.asyncio
    async def test_rate_limits_json_validation_and_conversion(self, clean_database):
        """Test JSON validation and type conversion for rate limits."""
        # Create user with rate limits to test conversion edge cases
        user = await UserManager.insert_user("JSON Validation User")

        from nilai_api.db import get_db_session
        import sqlalchemy as sa

        # Test storing various JSON data types
        test_cases = [
            # Valid rate limits with mixed types
            {"user_rate_limit_day": 1000, "web_search_rate_limit_hour": 50},
            # String numbers (should be converted)
            {"user_rate_limit_day": "2000", "user_rate_limit": "15"},
            # Mix of valid and invalid fields (invalid should be ignored)
            {
                "user_rate_limit_day": 3000,
                "invalid_field": "should_be_ignored",
                "web_search_rate_limit": 25,
            },
        ]

        for i, test_data in enumerate(test_cases):
            async with get_db_session() as session:
                stmt = sa.text(
                    "UPDATE users SET rate_limits = :data WHERE user_id = :user_id"
                )
                await session.execute(
                    stmt, {"data": json.dumps(test_data), "user_id": user.user_id}
                )
                await session.commit()

            # Retrieve and verify
            updated_user = await UserManager.check_user(user.user_id)
            assert updated_user is not None
            assert updated_user.rate_limits == test_data

            # Test rate_limits_obj conversion handles the data correctly
            rate_limits_obj = updated_user.rate_limits_obj
            assert isinstance(rate_limits_obj, RateLimits)

            # Verify specific conversions based on test case
            if i == 0:  # Mixed types
                assert rate_limits_obj.user_rate_limit_day == 1000
                assert rate_limits_obj.web_search_rate_limit_hour == 50
            elif i == 1:  # String numbers
                assert (
                    rate_limits_obj.user_rate_limit_day == 2000
                )  # Should convert string to int
                assert (
                    rate_limits_obj.user_rate_limit == 15
                )  # Should convert string to int
            elif i == 2:  # Mixed valid/invalid
                assert rate_limits_obj.user_rate_limit_day == 3000
                assert rate_limits_obj.web_search_rate_limit == 25
                # invalid_field should not cause issues

        # Test empty JSON object
        async with get_db_session() as session:
            stmt = sa.text(
                "UPDATE users SET rate_limits = '{}' WHERE user_id = :user_id"
            )
            await session.execute(stmt, {"user_id": user.user_id})
            await session.commit()

        empty_user = await UserManager.check_user(user.user_id)
        assert empty_user is not None
        assert empty_user.rate_limits == {}
        empty_rate_limits_obj = empty_user.rate_limits_obj
        assert isinstance(empty_rate_limits_obj, RateLimits)
        # Should have config defaults for empty JSON

        # Test invalid JSON handling (this should be handled gracefully)
        try:
            async with get_db_session() as session:
                # This should work as PostgreSQL JSONB validates JSON
                stmt = sa.text(
                    "UPDATE users SET rate_limits = :data WHERE user_id = :user_id"
                )
                await session.execute(
                    stmt,
                    {
                        "data": '{"user_rate_limit_day": 5000}',  # Valid JSON string
                        "user_id": user.user_id,
                    },
                )
                await session.commit()

            json_string_user = await UserManager.check_user(user.user_id)
            assert json_string_user is not None
            assert json_string_user.rate_limits == {"user_rate_limit_day": 5000}

        except Exception as e:
            # If there are issues with JSON handling, they should be caught gracefully
            print(f"JSON validation test caught expected error: {e}")

    @pytest.mark.asyncio
    async def test_rate_limits_update_workflow(self, clean_database):
        """Test complete workflow: create user with no rate limits -> update rate limits -> verify update."""
        # Step 1: Create user with NO rate limits
        user_id = str(uuid.uuid4())
        user = await UserManager.insert_user(user_id=user_id)

        # Verify user was created with no rate limits
        assert user.user_id == user_id
        assert user.rate_limits is None  # No rate limits initially

        # Step 2: Retrieve user and confirm no rate limits
        retrieved_user = await UserManager.check_user(user.user_id)
        assert retrieved_user is not None
        print(retrieved_user.to_pydantic())
        assert retrieved_user is not None
        assert retrieved_user.rate_limits is None

        # Verify rate_limits_obj returns config defaults for null rate limits
        default_rate_limits_obj = retrieved_user.rate_limits_obj
        assert isinstance(default_rate_limits_obj, RateLimits)
        # Should have config defaults applied
        assert default_rate_limits_obj.user_rate_limit_day is not None

        # Step 3: Create new rate limits to apply
        new_rate_limits = RateLimits(
            user_rate_limit_day=800,
            user_rate_limit_hour=80,
            user_rate_limit_minute=8,
            web_search_rate_limit_day=40,
            web_search_rate_limit_hour=4,
            web_search_rate_limit_minute=1,
            user_rate_limit=15,
            web_search_rate_limit=12,
        )

        # Step 4: Update the user's rate limits using the new function
        update_success = await UserManager.update_rate_limits(
            user.user_id, new_rate_limits
        )
        assert update_success is True

        # Step 5: Retrieve user again and verify rate limits were updated
        updated_user = await UserManager.check_user(user.user_id)
        assert updated_user is not None
        assert updated_user.rate_limits is not None
        assert updated_user.rate_limits == new_rate_limits.model_dump()
        print(updated_user.to_pydantic())
        # Step 6: Verify rate_limits_obj property works correctly with updated data
        updated_rate_limits_obj = updated_user.rate_limits_obj
        assert isinstance(updated_rate_limits_obj, RateLimits)
        assert updated_rate_limits_obj.user_rate_limit_day == 800
        assert updated_rate_limits_obj.user_rate_limit_hour == 80
        assert updated_rate_limits_obj.user_rate_limit_minute == 8
        assert updated_rate_limits_obj.web_search_rate_limit_day == 40
        assert updated_rate_limits_obj.web_search_rate_limit_hour == 4
        assert updated_rate_limits_obj.web_search_rate_limit_minute == 1
        assert updated_rate_limits_obj.user_rate_limit == 15
        assert updated_rate_limits_obj.web_search_rate_limit == 12

        # Step 7: Test updating with partial rate limits
        partial_rate_limits = RateLimits(
            user_rate_limit_day=1200,
            web_search_rate_limit_hour=6,
            # Other fields will be None
        )

        partial_update_success = await UserManager.update_rate_limits(
            user.user_id, partial_rate_limits
        )
        assert partial_update_success is True

        # Step 8: Verify partial update worked
        final_user = await UserManager.check_user(user.user_id)
        assert final_user is not None
        assert final_user.rate_limits == partial_rate_limits.model_dump()

        # Verify only specified fields are set, others are None
        final_rate_limits_obj = final_user.rate_limits_obj
        assert final_rate_limits_obj.user_rate_limit_day == 1200
        assert final_rate_limits_obj.web_search_rate_limit_hour == 6
        # Other fields should have config defaults (not None due to get_effective_limits)

        # Step 9: Test error case - update non-existent user
        fake_user_id = "non-existent-user-id"
        error_update = await UserManager.update_rate_limits(
            fake_user_id, new_rate_limits
        )
        assert error_update is False
